#!/usr/bin/env python3
"""
Fix TensorFlow execstack issue on WSL.

WSL kernels reject shared libraries that request an executable stack
(PT_GNU_STACK with PF_X flag = 0x7).  This script clears the PF_X bit
so that the flag becomes PF_R|PF_W (0x6).  This is safe — TensorFlow
does not actually need an executable stack.

Run this ONCE per new conda/mamba environment that has TensorFlow installed:

    conda activate lseqmdl        # or whichever env
    python scripts/fix_tf_execstack.py

After patching, verify with:

    python -c "import tensorflow as tf; print(tf.__version__)"
"""
import struct
import glob
import os
import sys


def find_tensorflow_dir():
    """Locate the tensorflow package directory."""
    env = os.environ.get("CONDA_PREFIX", sys.prefix)
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate = os.path.join(env, "lib", pyver, "site-packages", "tensorflow")
    if os.path.isdir(candidate):
        return candidate

    # Fallback: search sys.path
    for sp in sys.path:
        candidate = os.path.join(sp, "tensorflow")
        if os.path.isdir(candidate):
            return candidate

    return None


def patch_execstack(tf_dir):
    """Scan all .so files under tf_dir and clear PF_X from PT_GNU_STACK."""
    PT_GNU_STACK = 0x6474E551
    PF_X = 0x1

    patched = 0
    scanned = 0

    for so_path in glob.glob(os.path.join(tf_dir, "**", "*.so*"), recursive=True):
        try:
            with open(so_path, "r+b") as f:
                magic = f.read(4)
                if magic != b"\x7fELF":
                    continue
                scanned += 1

                # Only handle 64-bit ELF
                f.seek(4)
                ei_class = struct.unpack("B", f.read(1))[0]
                if ei_class != 2:
                    continue

                # Read program header table offset and count
                f.seek(32)
                e_phoff = struct.unpack("<Q", f.read(8))[0]
                f.seek(54)
                e_phentsize = struct.unpack("<H", f.read(2))[0]
                e_phnum = struct.unpack("<H", f.read(2))[0]

                for i in range(e_phnum):
                    off = e_phoff + i * e_phentsize
                    f.seek(off)
                    p_type = struct.unpack("<I", f.read(4))[0]
                    if p_type == PT_GNU_STACK:
                        p_flags = struct.unpack("<I", f.read(4))[0]
                        if p_flags & PF_X:
                            new_flags = p_flags & ~PF_X
                            f.seek(off + 4)
                            f.write(struct.pack("<I", new_flags))
                            name = os.path.basename(so_path)
                            print(f"  PATCHED: {name}  (flags 0x{p_flags:x} -> 0x{new_flags:x})")
                            patched += 1
                        break
        except (PermissionError, OSError) as exc:
            print(f"  SKIP: {os.path.basename(so_path)} ({exc})")

    return scanned, patched


def main():
    tf_dir = find_tensorflow_dir()
    if tf_dir is None:
        print("ERROR: Could not find tensorflow package directory.")
        print(f"  CONDA_PREFIX = {os.environ.get('CONDA_PREFIX', '(not set)')}")
        print(f"  sys.prefix   = {sys.prefix}")
        sys.exit(1)

    print(f"Scanning: {tf_dir}")
    scanned, patched = patch_execstack(tf_dir)
    print(f"\nDone. Scanned {scanned} ELF files, patched {patched}.")

    if patched > 0:
        print('\nVerify with:  python -c "import tensorflow as tf; print(tf.__version__)"')
    else:
        print("\nNo files needed patching (already fixed or no exec-stack flags found).")


if __name__ == "__main__":
    main()

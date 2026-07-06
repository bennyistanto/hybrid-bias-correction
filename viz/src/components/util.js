// Shared constants and helpers for the dashboard pages.

export const STAGES = ["LS", "LSEQM", "LSEQM+DL"];
export const STAGE_KEY = {"LS": "ls", "LSEQM": "lseqm", "LSEQM+DL": "lseqmdl"};

// Stage colours: LS a muted baseline, the two corrected stages warmer/cooler.
export const STAGE_COLORS = {"LS": "#9aa7b0", "LSEQM": "#e0913a", "LSEQM+DL": "#2f7d9e"};

export const REGIONS = [
  "Sumatra", "Kalimantan", "Sulawesi", "Jawa",
  "Bali Nusa Tenggara", "Maluku", "Papua",
];

// Colours for the window-diagnostic reference series (shared across pages).
export const WINDOW_COLORS = {
  "vs CPC-UNI (UTC label)": "#c0392b",
  "vs BMKG (validation)": "#2980b9",
  "vs CPC-UNI (relabelled +1 day)": "#e08e0b",
};

// Number formatter: 3 decimals for small magnitudes, else 2. null -> hyphen.
export function fmt(v) {
  return v == null ? "-" : (Math.abs(v) < 0.1 && v !== 0 ? v.toFixed(3) : v.toFixed(2));
}

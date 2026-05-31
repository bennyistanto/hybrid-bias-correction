# Contributing

Thank you for your interest in `hybrid-bias-correction`. The framework is in
public release and welcomes bug reports, documentation improvements, and
contributions that extend its reach to other regions or reference products.

This guide covers: reporting bugs, asking questions, proposing changes,
running tests, and the development workflow.

## Code of conduct

By participating, you agree to abide by the [Contributor Covenant Code of
Conduct](CODE_OF_CONDUCT.md). Reports of unacceptable behaviour can be sent
to the maintainer (contact in `CITATION.cff`).

## Reporting bugs

Open a [GitHub issue](https://github.com/bennyistanto/hybrid-bias-correction/issues).
Please include:

- A short title that summarises the problem.
- The exact command or notebook cell that produced the error.
- The full traceback or error log (a code block is fine).
- Your platform: OS, Python version, whether you are running locally or on
  Colab, and which `requirements.txt` / `environment.yml` snapshot you
  installed.
- A pointer to the input data or a minimal reproducer if the bug depends on
  data.

The bundled Bali example dataset (`data/example_bali/`) is the preferred test
case for reproducers: it is small, public, and the maintainer can run it.

## Asking questions

For usage questions that are not bugs, open a [GitHub Discussion](https://github.com/bennyistanto/hybrid-bias-correction/discussions)
rather than an issue. Searchable for future readers.

## Proposing changes

For small fixes (documentation typos, broken links, minor code issues), open
a pull request directly. For larger changes (new pipeline stage, new
reference product support, new validation metric, breaking config change),
open an issue first to discuss scope; this avoids duplicated work.

A typical contribution flow:

1. Fork the repository and create a feature branch off `main`.
2. Make your change. Keep changes focused: one concern per pull request.
3. Update the relevant documentation page under `docs/` if behaviour or
   interface changed.
4. Run the tests (see below) and confirm they still pass.
5. Open the pull request against `main`, referencing any related issue.

## Development setup

Clone the repository and create a Python environment with the bundled
dependencies:

```bash
git clone https://github.com/bennyistanto/hybrid-bias-correction.git
cd hybrid-bias-correction
conda env create -f environment.yml
conda activate hybrid-bias-correction
# or, with pip:
python -m pip install -r requirements.txt
```

The src/ modules are designed to be importable directly without installation
(`from src import bias_correction`). Notebooks read paths from `config.yml`;
the `config_bali.yml` template is wired to the bundled Bali example.

## Running tests

The smoke test exercises the pipeline on tiny synthetic arrays without
needing the data archive:

```bash
python -m pytest tests/ -v
```

If you add a new pipeline stage or change a stage interface, please add a
matching test that operates on synthetic input so the suite continues to
run on free CI runners.

## Coding conventions

- Python style follows PEP 8 with line length 100. The codebase is mostly
  scientific/numerical; clarity wins over abstraction.
- Functions that touch the file system or load NetCDFs go in `src/io.py`.
- Functions that change the *meaning* of a metric or correction stage must
  preserve backwards-compatibility for `config.yml` keys; if a breaking
  change is required, bump the major version in `CHANGELOG.md` and document
  the migration in the relevant docs page.
- Notebook cells should call functions from `src/`; long inline code blocks
  in notebooks make the science harder to audit. The notebooks are a
  narrative interface, not the implementation.

## Documentation

The documentation is a Quarto site rendered from `docs/`. To preview locally:

```bash
quarto preview docs
```

Documentation changes accompany code changes in the same pull request when
they affect behaviour or the user-facing interface.

## Releases

Releases follow [DateVer](https://github.com/datever/datever) (`vYYYY.MM`).
The maintainer cuts releases; tagged releases are auto-archived on Zenodo
via the GitHub--Zenodo integration. See `CHANGELOG.md` for release notes.

## Attribution

By contributing, you agree that your contributions will be licensed under
the same Mozilla Public License 2.0 that covers the rest of the project.
Co-authors are added to `CITATION.cff` for substantive contributions
(method, large code areas, documentation refactor, supervisor-level
intellectual input).

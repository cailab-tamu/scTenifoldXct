# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Modern `pyproject.toml` packaging (replaces `setup.py`).
- Console scripts: `sctenifoldxct` and `sctenifoldxct-merge`.
- `scTenifoldXct.cli` module.
- Regression / characterization test suite (`tests/test_regression.py`).
- Public accessors on `scTenifoldXct`: `cell_names`, `genes`, `cell_data`,
  `get_data_arr()`, and package-level `__all__`.
- Documentation site (mkdocs) and `CHANGELOG.md`.

### Changed
- Minimum supported Python is now **3.10**; CI tests 3.10–3.12.
- Dependency declarations now carry tested lower bounds; `python_igraph`
  renamed to `igraph`.
- Package data now loaded via `importlib.resources` instead of the
  deprecated `pkg_resources`.
- All `print()` calls converted to the `logging` module. Output now
  requires a configured logging handler (e.g. `logging.basicConfig()`);
  the CLI configures this automatically.
- `merge_scTenifoldXct` now uses the public API instead of reaching into
  private attributes of `scTenifoldXct` instances.
- `environment.yml` slimmed to actual runtime + dev dependencies.
- README updated: PyPI install, CLI usage, corrected tutorial/Docker paths.
- Dockerfile: Python 3.10-slim, installs via `pyproject.toml`, fixed
  `adduser` flag, copies `data/`.

### Fixed
- `stat.py`: `('dist' or 'correspondence')` column guard always evaluated
  to only checking `'dist'`; now validates both columns.
- `core.py` `_build_w`: receptor metrics are now indexed with the receptor
  gene order (`_genes[receptor]`) — correct-by-construction; no change to
  numerical results.
- `visualization.py`: `edge_width_scale` was passed positionally as the
  `g1` argument in `plot_pcNet_method`; now passed as a keyword.
- `cko.py`: gene names are upper-cased to match the package/database
  convention, and the input AnnData is copied instead of mutated in place.

## [0.1.0] - 2026-05-15

Initial PyPI release.

# scTenifoldXct

A semi-supervised method for predicting cell-cell interactions and mapping
cellular communication graphs via manifold learning.

[Paper (Cell Systems, 2023)](https://doi.org/10.1016/j.cels.2023.01.004) ·
[Source](https://github.com/cailab-tamu/scTenifoldXct) ·
[Hosted demo](https://sctenifold.streamlit.app/)

**New:** run scTenifoldXct from your browser with no code — `pip install "scTenifoldXct[web]"` then
`sctenifoldxct-ui`. See [Web UI](usage.md#web-ui).

## Overview

scTenifoldXct builds gene regulatory networks for a sender and a receiver cell
type, aligns them onto a shared low-dimensional manifold, and scores
ligand–receptor pairs by their distance in that embedding. A non-parametric
test identifies significantly enriched interactions; a two-sample mode
(`merge_scTenifoldXct`) detects interactions that differ between conditions.

- **Installation** — see [Installation](installation.md)
- **Quick start** (Python and CLI) — see [Usage](usage.md)
- **API reference** — see [API reference](api.md)

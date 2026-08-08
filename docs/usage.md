# Usage

## Python

scTenifoldXct logs progress through the standard `logging` module. Configure a
handler to see messages when `verbose=True`:

```python
import logging
import scanpy as sc
import scTenifoldXct as st

logging.basicConfig(level=logging.INFO, format="%(message)s")

adata = sc.read_h5ad("data/adata_short_example.h5ad")
xct = st.scTenifoldXct(
    data=adata,
    source_celltype="Inflam. FIB",
    target_celltype="Inflam. DC",
    obs_label="ident",
    rebuild_GRN=True,
    GRN_file_dir="Net_example_dev",
    verbose=True,
    n_cpus=-1,
)
emb = xct.get_embeds(train=True)
xct_pairs = xct.null_test()
print(xct_pairs)
```

### Two-sample differential analysis

```python
from scTenifoldXct import merge_scTenifoldXct

merged = merge_scTenifoldXct(xct_condition_A, xct_condition_B)
emb = merged.get_embeds(train=True)
merged.nn_aligned_diff(emb)
diff_pairs = merged.chi2_diff_test()
```

## Command line

Two console scripts are installed with the package:

```shell
# single-sample interaction analysis
sctenifoldxct data/adata_short_example.h5ad --rebuild \
    -s "Inflam. FIB" -r "Inflam. DC" --n_cpus 8 -v

# two-sample differential interaction analysis
sctenifoldxct-merge data/adata_merge_example.h5ad NormalvsTumor N T \
    --rebuild -s "B cells" -r "Fibroblasts" --n_cpus 8 -v
```

Run `sctenifoldxct --help` or `sctenifoldxct-merge --help` for all options.

## Web UI

Prefer clicking over scripting? Install the `web` extra and launch a local, point-and-click
interface — no Python knowledge required:

```shell
pip install "scTenifoldXct[web]"
sctenifoldxct-ui
# opens http://127.0.0.1:8000
```

From a git checkout, the UI's "Use bundled example dataset" button loads
`data/adata_short_example.h5ad` directly, so you can go from install to results in under a minute.
Otherwise, upload your own `.h5ad`. Pick a cell-metadata column plus sender/receiver cell types,
tune the analysis options if you like, and hit **Run analysis** — ranked ligand-receptor pairs show
up in-browser with a CSV download for the full list. Everything runs locally; no data leaves your
machine. Run `sctenifoldxct-ui --help` for host/port/GRN-cache options.

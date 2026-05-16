"""
Command-line entry points for scTenifoldXct.

Installed as console scripts via pyproject.toml:
  sctenifoldxct        — single-sample interaction analysis
  sctenifoldxct-merge  — two-sample differential interaction analysis

These replicate the python -m scTenifoldXct.core / .merge interface.
"""
from __future__ import annotations

import argparse
import logging


def _core_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sctenifoldxct",
        description="scTenifoldXct: single-sample cell-cell interaction analysis.",
    )
    p.add_argument("file", type=str, help="Path to log-normalised AnnData (.h5ad)")
    p.add_argument("-w", "--workdir", type=str, default="xct_results", help="Output directory")
    p.add_argument("-o", "--output", type=str, default="xct_enriched", help="Output file stem")
    p.add_argument("-s", "--sender", type=str, default="cell_A", help="Sender cell type label")
    p.add_argument("-r", "--receiver", type=str, default="cell_B", help="Receiver cell type label")
    p.add_argument("-l", "--label", type=str, default="ident", help="obs column with cell-type labels")
    p.add_argument("--n_cpus", type=int, default=-1, help="CPUs for GRN construction (-1 = all)")
    p.add_argument("-v", "--verbose", action="store_true")
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--rebuild", dest="rebuild", action="store_true")
    grp.add_argument("--no-rebuild", dest="rebuild", action="store_false")
    p.set_defaults(rebuild=True)
    # hidden evaluation flags
    p.add_argument("--eva", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--n_sample", type=int, default=100, help=argparse.SUPPRESS)
    p.add_argument("--n_feature", type=int, default=3000, help=argparse.SUPPRESS)
    return p


def _merge_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sctenifoldxct-merge",
        description="scTenifoldXct: two-sample differential interaction analysis.",
    )
    p.add_argument("file", type=str, help="Path to log-normalised AnnData (.h5ad)")
    p.add_argument("cond_label", type=str, help="obs column distinguishing the two conditions")
    p.add_argument("cond_WT", type=str, help="Reference condition label")
    p.add_argument("cond_KO", type=str, help="Comparison condition label")
    p.add_argument("-w", "--workdir", type=str, default="xct_results", help="Output directory")
    p.add_argument("-o", "--output", type=str, default="xct_enriched_diff", help="Output file stem")
    p.add_argument("-s", "--sender", type=str, default="cell_A", help="Sender cell type label")
    p.add_argument("-r", "--receiver", type=str, default="cell_B", help="Receiver cell type label")
    p.add_argument("-l", "--label", type=str, default="ident", help="obs column with cell-type labels")
    p.add_argument("--n_cpus", type=int, default=-1, help="CPUs for GRN construction (-1 = all)")
    p.add_argument("-v", "--verbose", action="store_true")
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--rebuild", dest="rebuild", action="store_true")
    grp.add_argument("--no-rebuild", dest="rebuild", action="store_false")
    p.set_defaults(rebuild=True)
    p.add_argument("--eva", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--n_sample", type=int, default=100, help=argparse.SUPPRESS)
    p.add_argument("--n_feature", type=int, default=3000, help=argparse.SUPPRESS)
    return p


def xct() -> None:
    """Entry point for single-sample analysis (sctenifoldxct)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from scTenifoldXct.core import main
    main(_core_parser().parse_args())


def xct_merge() -> None:
    """Entry point for two-sample differential analysis (sctenifoldxct-merge)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from scTenifoldXct.merge import main
    main(_merge_parser().parse_args())

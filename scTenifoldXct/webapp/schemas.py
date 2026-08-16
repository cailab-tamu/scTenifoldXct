"""Pydantic request/response models for the scTenifoldXct web API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    n_genes: int
    n_cells: int
    obs_labels: dict[str, list[str]] = Field(
        default_factory=dict,
        description="obs column name -> sorted unique values, for columns usable as cell-type labels",
    )
    prebuilt_grn: bool = Field(
        False,
        description="whether a precomputed gene regulatory network is bundled for this dataset "
        "(skips the slow network-reconstruction step on first run)",
    )


class JobCreate(BaseModel):
    dataset_id: str
    source_celltype: str = Field(..., description="sender cell type (must appear in obs_label column)")
    target_celltype: str = Field(..., description="receiver cell type (must appear in obs_label column)")
    obs_label: str = "ident"
    query_db: Literal["comb", "pairs"] | None = Field(
        None, description="restrict candidate pairs to the ligand-receptor database: None, 'comb', or 'pairs'"
    )
    alpha: float = Field(0.5, ge=0, le=1, description="mean/variance weighting for the correspondence score")
    mu: float = Field(1.0, gt=0, description="scale factor applied to the ligand-receptor correspondence block")
    scale_w: bool = True
    n_dim: int = Field(3, ge=1, le=50, description="embedding dimensionality for manifold alignment")
    n_steps: int = Field(1000, ge=1, le=20000, description="training steps for the manifold alignment network")
    lr: float = Field(0.01, gt=0, description="training learning rate")
    dist_metric: str = "euclidean"
    test_method: Literal["null", "chi2"] = Field("null", description="enrichment test used to rank pairs")
    pval: float = Field(0.05, gt=0, lt=1)
    filter_zeros: bool = Field(True, description="null test only: drop zero-correspondence background pairs")
    dof: int = Field(1, ge=1, description="chi2 test only: degrees of freedom")
    fdr: bool = Field(True, description="chi2 test only: apply Benjamini-Hochberg FDR correction")
    rebuild_grn: bool = Field(
        True, description="rebuild the gene regulatory networks instead of reusing a cached/prebuilt one"
    )
    n_cpus: int = Field(-1, ge=-1, le=32, description="CPUs for network reconstruction (-1 = all)")
    seed: int | None = None

    @model_validator(mode="after")
    def _celltypes_differ(self):
        if self.source_celltype == self.target_celltype:
            raise ValueError("source_celltype and target_celltype must differ")
        return self


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    stage: str
    error: str | None = None


class PairRow(BaseModel):
    pair: str
    ligand: str
    receptor: str
    dist: float
    correspondence: float | None = None
    p_val: float | None = None
    q_val: float | None = None
    FC: float | None = None
    enriched_rank: int | None = None


class JobResult(BaseModel):
    job_id: str
    source_celltype: str
    target_celltype: str
    test_method: str
    rows: list[PairRow]

import pytest
import numpy as np
import scanpy as sc
from scTenifoldXct.core import scTenifoldXct


@pytest.fixture(scope="session")
def ada_test():
    ada = sc.datasets.paul15()[:, -500:].copy()  # raw counts
    del ada.uns
    ada.layers['raw'] = np.asarray(ada.X, dtype=int)
    sc.pp.log1p(ada)
    ada.layers['log1p'] = ada.X.copy()
    return ada


# small dataset
@pytest.fixture(scope="session")
def xct_test(ada_test):
    return scTenifoldXct(data=ada_test, 
                        source_celltype='14Mo',
                        target_celltype='15Mo',
                        obs_label="paul15_clusters",
                        rebuild_GRN=True, 
                        GRN_file_dir='./Net_for_Test',
                        verbose=True, 
                        n_cpus=2,
                        )


@pytest.fixture(scope="session")
def xct_test_r(ada_test):
    return scTenifoldXct(data=ada_test, 
                        source_celltype='15Mo',
                        target_celltype='14Mo',
                        obs_label="paul15_clusters",
                        rebuild_GRN=False, 
                        GRN_file_dir='./Net_for_Test',
                        verbose=True, 
                        )

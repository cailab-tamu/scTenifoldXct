import pytest
from scTenifoldXct.core import GRN


def test_xct_obj_attrs(xct_test):
    assert isinstance(xct_test.net_A, GRN), "net_A should be a GRN object"
    assert isinstance(xct_test.net_B, GRN), "net_B should be a GRN object"
    assert xct_test.net_B.shape == xct_test.net_A.shape

import logging

from .version import __version__  # noqa: F401
from scTenifoldXct.core import scTenifoldXct
from scTenifoldXct.visualization import get_Xct_pairs, plot_XNet
from scTenifoldXct.merge import merge_scTenifoldXct

logging.getLogger(__name__).addHandler(logging.NullHandler())
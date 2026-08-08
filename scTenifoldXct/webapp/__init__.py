"""Local web UI for scTenifoldXct.

Install with the ``web`` extra and launch:

    pip install "scTenifoldXct[web]"
    sctenifoldxct-ui

Serves a single-page app (static/) backed by a small FastAPI JSON API
(main.py) that runs the scTenifoldXct workflow (scTenifoldXct/core.py) as
background jobs.
"""

from .main import create_app

__all__ = ["create_app"]

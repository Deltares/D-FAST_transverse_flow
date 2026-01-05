# -*- coding: utf-8 -*-

from typing import Any, List

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("d-fast-traverse-flow")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

__all__: List[Any] = []

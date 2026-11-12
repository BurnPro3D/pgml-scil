"""Opens FIRETEC Zarr stores, from disk or through a Pelican federation."""

from collections.abc import Mapping
from typing import Any

import xarray as xr


def is_remote(uri: str) -> bool:
    """True for an fsspec URL such as pelican://, osdf:// or https://."""
    return "://" in uri


def open_firetec_store(
    uri: str, storage_options: Mapping[str, Any] | None = None
) -> xr.Dataset:
    """Open a FIRETEC Zarr store without loading its arrays.

    ``uri`` is a path on disk or an fsspec URL. PelicanFS adds two schemes:

    - ``pelican://<federation>/<namespace>/<path>.zarr`` for any federation
    - ``osdf:///<namespace>/<path>.zarr`` for the Open Science Data Federation

    Pelican serves objects from the federation's caches unless told otherwise.
    ``storage_options`` go to the URL's filesystem: for Pelican that is
    ``PelicanFileSystem``, which accepts for example ``{"direct_reads": True}``
    to read from the origin instead of a cache. They are ignored for a path on
    disk, so a configuration switches between disk and Pelican by changing the
    URI alone.
    """
    options = dict(storage_options or {}) if is_remote(uri) else {}
    return xr.open_dataset(uri, engine="zarr", storage_options=options or None)

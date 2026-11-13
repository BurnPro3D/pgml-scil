"""Opens FIRETEC Zarr stores from disk, from S3, or through a Pelican federation,
and copies run folders to such places."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import fsspec
import xarray as xr


def url_scheme(uri: str) -> str | None:
    """The scheme of an fsspec URL (``s3``, ``osdf``, ``https``, ...); None for a path on disk."""
    scheme, separator, _ = uri.partition("://")
    return scheme if separator else None


def filesystem_options(
    uri: str, storage_options: Mapping[str, Mapping[str, Any] | None] | None
) -> dict[str, Any]:
    """The options for the filesystem that reads ``uri``.

    ``storage_options`` maps a URL scheme to the options of its filesystem, so
    one configuration can hold the options of every service its stores are on:

        {"s3": {"endpoint_url": "https://s3-west.nrp-nautilus.io", "anon": True},
         "osdf": {"direct_reads": True}}

    A path on disk, or a URL whose scheme has no entry, gets no options.
    """
    storage_options = storage_options or {}
    for scheme, options in storage_options.items():
        if options is not None and not isinstance(options, Mapping):
            raise ValueError(
                "storage_options maps a URL scheme (s3, osdf, pelican, ...) to the options "
                f"of its filesystem, but its entry {scheme!r} is {options!r}"
            )
    scheme = url_scheme(uri)
    if scheme is None:
        return {}
    return dict(storage_options.get(scheme) or {})


def open_firetec_store(
    uri: str,
    storage_options: Mapping[str, Mapping[str, Any] | None] | None = None,
    zarr_format: int | str | None = None,
) -> xr.Dataset:
    """Open a FIRETEC Zarr store without loading its arrays.

    ``uri`` is a path on disk or an fsspec URL:

    - ``s3://<bucket>/<path>.zarr`` for S3 or an S3-compatible service (s3fs)
    - ``pelican://<federation>/<namespace>/<path>.zarr`` for any Pelican federation
    - ``osdf:///<namespace>/<path>.zarr`` for the Open Science Data Federation

    The URL's filesystem gets its scheme's entry of ``storage_options`` (see
    `filesystem_options`), so a configuration switches between disk, S3 and
    Pelican by changing the URI alone.

    ``zarr_format`` is 2 or 3, or None to detect it. Detection looks for the
    format 3 metadata first, and fails on an S3 bucket that does not allow
    listing: such a bucket answers "access denied" for a missing object, not
    "not found". The format may be given as text, which is how a command-line
    override of a null configuration value arrives.
    """
    options = filesystem_options(uri, storage_options)
    return xr.open_dataset(
        uri,
        engine="zarr",
        storage_options=options or None,
        zarr_format=None if zarr_format is None else int(zarr_format),
    )


def upload_folder(
    folder: str | Path,
    url: str,
    storage_options: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> str:
    """Copy ``folder`` to ``<url>/<folder name>`` and return that URL.

    ``url`` is a path or an fsspec URL such as
    ``s3://ndp-cat1/firetec/pgml/runs``, and its filesystem gets its scheme's
    entry of ``storage_options``, except ``anon``: nobody writes anonymously,
    so the same options that read a public bucket without credentials write to
    it with the credentials in the environment (for S3: AWS_ACCESS_KEY_ID and
    AWS_SECRET_ACCESS_KEY). Uploading the same folder again overwrites its files.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    destination = f"{url.rstrip('/')}/{folder.name}"
    options = filesystem_options(url, storage_options)
    options.pop("anon", None)
    filesystem, path = fsspec.core.url_to_fs(destination, **options)
    # The trailing slash copies the folder's contents to the destination even
    # when it exists already, instead of into a new folder inside it.
    filesystem.put(f"{folder}/", path, recursive=True)
    return destination

from pathlib import Path

import pytest
import xarray as xr

from conftest import NUM_FRAMES
from firetec_pgml.store import is_remote, open_firetec_store


def test_opens_a_store_from_a_path(firetec_store):
    with open_firetec_store(firetec_store) as store:
        assert store.sizes["time"] == NUM_FRAMES
        assert store.attrs["firetec_parameters"]["zu"]["value"] == 25.0


def test_opens_a_store_from_a_url(firetec_store):
    """A file:// URL takes the fsspec route that pelican:// and osdf:// URLs take."""
    with open_firetec_store(Path(firetec_store).as_uri()) as store:
        assert store.sizes["time"] == NUM_FRAMES


def test_storage_options_are_passed_on_for_a_url(monkeypatch):
    calls = []
    monkeypatch.setattr(xr, "open_dataset", lambda uri, **options: calls.append((uri, options)))
    open_firetec_store("osdf:///namespace/run.zarr", {"direct_reads": True})
    assert calls == [
        ("osdf:///namespace/run.zarr", {"engine": "zarr", "storage_options": {"direct_reads": True}})
    ]


def test_storage_options_are_ignored_for_a_path(firetec_store):
    """Zarr refuses storage options for a path, so they must not be passed on."""
    with open_firetec_store(firetec_store, {"direct_reads": True}) as store:
        assert store.sizes["time"] == NUM_FRAMES


@pytest.mark.parametrize(
    "uri, remote",
    [
        ("/data/run.zarr", False),
        ("runs/run.zarr", False),
        ("osdf:///namespace/run.zarr", True),
        ("pelican://federation.org/namespace/run.zarr", True),
    ],
)
def test_urls_are_told_apart_from_paths(uri, remote):
    assert is_remote(uri) is remote

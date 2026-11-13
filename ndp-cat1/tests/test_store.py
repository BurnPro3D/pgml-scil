from pathlib import Path

import pytest
import xarray as xr

from conftest import INITIAL_FUEL, NUM_FRAMES
from firetec_pgml.store import filesystem_options, open_firetec_store, url_scheme

S3_OPTIONS = {"endpoint_url": "https://s3.example.org", "anon": True}
STORAGE_OPTIONS = {"s3": S3_OPTIONS, "osdf": {"direct_reads": True}}


def test_opens_a_store_from_a_path(firetec_store):
    with open_firetec_store(firetec_store) as store:
        assert store.sizes["time"] == NUM_FRAMES
        assert store.attrs["firetec_parameters"]["zu"]["value"] == 25.0


def test_opens_a_store_from_a_url(firetec_store):
    """A file:// URL takes the fsspec route that s3://, pelican:// and osdf:// URLs take."""
    with open_firetec_store(Path(firetec_store).as_uri(), STORAGE_OPTIONS) as store:
        assert store.sizes["time"] == NUM_FRAMES


def test_storage_options_are_ignored_for_a_path(firetec_store):
    """Zarr refuses storage options for a path, so they must not be passed on."""
    with open_firetec_store(firetec_store, STORAGE_OPTIONS, zarr_format=2) as store:
        assert store.sizes["time"] == NUM_FRAMES


def test_the_zarr_format_opens_a_store_in_a_bucket_that_denies_listing(listing_denied_store):
    """Detecting the format would ask for format 3 metadata, which such a bucket denies."""
    with open_firetec_store(listing_denied_store, zarr_format=2) as store:
        assert store.sizes["time"] == NUM_FRAMES
        assert float(store["rhoFuel"].max()) == INITIAL_FUEL


def test_the_zarr_format_may_be_given_as_text(firetec_store):
    """A command-line override of a null configuration value arrives as text."""
    with open_firetec_store(firetec_store, zarr_format="2") as store:
        assert store.sizes["time"] == NUM_FRAMES


def test_options_and_format_are_passed_on_for_a_url(monkeypatch):
    calls = []
    monkeypatch.setattr(xr, "open_dataset", lambda uri, **options: calls.append((uri, options)))
    open_firetec_store("s3://bucket/run.zarr", STORAGE_OPTIONS, zarr_format=2)
    assert calls == [
        (
            "s3://bucket/run.zarr",
            {"engine": "zarr", "storage_options": S3_OPTIONS, "zarr_format": 2},
        )
    ]


@pytest.mark.parametrize(
    "uri, options",
    [
        ("s3://bucket/run.zarr", S3_OPTIONS),
        ("osdf:///namespace/run.zarr", {"direct_reads": True}),
        ("pelican://federation.org/namespace/run.zarr", {}),
        ("/data/run.zarr", {}),
    ],
)
def test_a_url_gets_the_options_of_its_scheme(uri, options):
    assert filesystem_options(uri, STORAGE_OPTIONS) == options


def test_a_scheme_may_be_listed_without_options():
    assert filesystem_options("osdf:///namespace/run.zarr", {"osdf": None}) == {}


def test_options_not_keyed_by_scheme_are_refused():
    """The options of one filesystem, given without naming its scheme."""
    with pytest.raises(ValueError, match="direct_reads"):
        filesystem_options("osdf:///namespace/run.zarr", {"direct_reads": True})


@pytest.mark.parametrize(
    "uri, scheme",
    [
        ("/data/run.zarr", None),
        ("runs/run.zarr", None),
        ("s3://bucket/run.zarr", "s3"),
        ("osdf:///namespace/run.zarr", "osdf"),
        ("pelican://federation.org/namespace/run.zarr", "pelican"),
        ("https://s3-west.nrp-nautilus.io/bucket/run.zarr", "https"),
    ],
)
def test_urls_are_told_apart_from_paths(uri, scheme):
    assert url_scheme(uri) == scheme

from collections.abc import Iterator

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from firetec_pgml import store
from firetec_pgml.store import upload_folder
from train import train
from upload import upload


@pytest.fixture
def bucket(tmp_path) -> Iterator[str]:
    """URL of an empty in-memory bucket."""
    url = f"memory://{tmp_path.name}-bucket/runs"
    yield url
    if MemoryFileSystem().exists(url):
        MemoryFileSystem().rm(url, recursive=True)


@pytest.fixture
def run_folder(tmp_path):
    folder = tmp_path / "20261113_120000_test"
    (folder / "evaluation" / "run").mkdir(parents=True)
    (folder / "config.yaml").write_text("stores: []\n")
    (folder / "evaluation" / "run" / "forecast_metrics.csv").write_text("frame\n6\n")
    return folder


def files_under(url: str) -> list[str]:
    filesystem, path = fsspec.core.url_to_fs(url)
    return sorted(found.removeprefix(path.rstrip("/") + "/") for found in filesystem.find(path))


def test_a_folder_is_copied_under_its_own_name(run_folder, bucket):
    destination = upload_folder(run_folder, bucket)
    assert destination == f"{bucket}/{run_folder.name}"
    assert files_under(destination) == ["config.yaml", "evaluation/run/forecast_metrics.csv"]


def test_uploading_again_overwrites_instead_of_nesting(run_folder, bucket):
    upload_folder(run_folder, bucket)
    (run_folder / "config.yaml").write_text("stores: [again]\n")
    destination = upload_folder(run_folder, bucket)
    assert files_under(destination) == ["config.yaml", "evaluation/run/forecast_metrics.csv"]
    filesystem, path = fsspec.core.url_to_fs(f"{destination}/config.yaml")
    assert filesystem.cat_file(path) == b"stores: [again]\n"


def test_the_upload_uses_the_schemes_options_without_anon(run_folder, monkeypatch):
    calls = []

    def url_to_fs(url, **options):
        calls.append((url, options))
        return MemoryFileSystem(), f"/{run_folder.parent.name}/{run_folder.name}"

    monkeypatch.setattr(store.fsspec.core, "url_to_fs", url_to_fs)
    storage_options = {"s3": {"endpoint_url": "https://s3.example.org", "anon": True}}
    upload_folder(run_folder, "s3://bucket/runs/", storage_options)
    assert calls == [
        (f"s3://bucket/runs/{run_folder.name}", {"endpoint_url": "https://s3.example.org"})
    ]
    MemoryFileSystem().rm(f"/{run_folder.parent.name}", recursive=True)


def test_a_missing_folder_is_refused(tmp_path, bucket):
    with pytest.raises(FileNotFoundError):
        upload_folder(tmp_path / "missing", bucket)


def test_a_trained_run_is_uploaded_with_its_files(config, bucket):
    run_dir = train(config)
    destination = upload(run_dir, bucket)
    assert {"config.yaml", "training.log", "history.csv", "best_model.pth"} <= set(
        files_under(destination)
    )

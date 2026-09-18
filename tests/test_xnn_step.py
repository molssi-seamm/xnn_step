#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the `xnn_step` package: the model-chemistry / MDI-engine provider."""

from types import SimpleNamespace

import pytest  # noqa: F401
import xnn_step
from xnn_step import XnnStep


def test_construction():
    """Just create an object and test its type."""
    result = xnn_step.Xnn()
    assert str(type(result)) == "<class 'xnn_step.xnn.Xnn'>"


@pytest.fixture()
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "water_mace.pt").write_bytes(b"x")
    (d / "salt@2026.pt").write_bytes(b"x")  # reserved character in the stem
    (d / "notes.txt").write_text("not a model")
    return d


@pytest.fixture()
def ini(tmp_path, models_dir):
    """A root with an xnn.ini pointing at the temporary model directory."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "xnn.ini").write_text(
        "[local]\n"
        "installation = conda\n"
        "conda = /opt/conda/bin/conda\n"
        "conda-environment = seamm-xnn\n"
        "code = xnn\n"
        "device = cpu\n"
        f"models = {models_dir}\n"
        "pattern = *.pt\n"
    )
    return root


def test_model_name_replaces_reserved_characters():
    assert XnnStep.model_name("/a/b/salt@2026:x/y|z.pt") == "y-z"
    assert XnnStep.model_name("water_mace.pt") == "water_mace"


def test_available_models_from_config(models_dir):
    config = {"models": str(models_dir), "pattern": "*.pt"}
    models = XnnStep.available_models(config)
    assert sorted(models) == ["salt-2026", "water_mace"]
    assert models["water_mace"] == models_dir / "water_mace.pt"


def test_available_models_missing_directory_is_empty(tmp_path):
    config = {"models": str(tmp_path / "nowhere"), "pattern": "*.pt"}
    assert XnnStep.available_models(config) == {}


def test_available_models_root_substitution(tmp_path, models_dir, monkeypatch):
    monkeypatch.setattr(XnnStep, "seamm_root", staticmethod(lambda: tmp_path))
    config = {"models": "{root}/models", "pattern": "*.pt"}
    assert "water_mace" in XnnStep.available_models(config)


def test_get_model_chemistry_options(models_dir, monkeypatch):
    config = {"models": str(models_dir), "pattern": "*.pt"}
    monkeypatch.setattr(XnnStep, "_local_config", classmethod(lambda cls: config))
    options = XnnStep.get_model_chemistry_options()
    assert set(options) == {"salt-2026", "water_mace"}
    entry = options["water_mace"]
    assert entry["model_chemistry"] == "xnn:MLFF@water_mace"
    assert entry["type"] == "MLFF"
    assert entry["mdi_capable"] is True
    assert entry["periodic_mdi"] is True
    assert entry["mdi_method_arg"] == "water_mace"
    assert entry["path"].endswith("water_mace.pt")
    # The filters never exclude anything for an MLFF.
    assert set(XnnStep.get_model_chemistry_options(periodic_only=True)) == set(options)


def test_get_executor_config(ini):
    executor = SimpleNamespace(name="local")
    config = XnnStep.get_executor_config(executor, {"root": str(ini)})
    assert config["conda-environment"] == "seamm-xnn"
    assert config["device"] == "cpu"
    assert "version" in config


def test_get_mdi_engine_command_conda(ini, models_dir, monkeypatch):
    monkeypatch.setattr(XnnStep, "thread_count", classmethod(lambda cls: None))
    executor = SimpleNamespace(name="local")
    argv = XnnStep.get_mdi_engine_command(
        executor,
        {"root": str(ini)},
        method="water_mace",
        port=8021,
        hostname="localhost",
        charge=0,
        multiplicity=1,
        n_atoms=3,
    )
    assert argv[:5] == [
        "/opt/conda/bin/conda",
        "run",
        "--live-stream",
        "-n",
        "seamm-xnn",
    ]
    assert argv[5:7] == ["xnn", "mdi"]
    i = argv.index("--ckpt")
    assert argv[i + 1] == str(models_dir / "water_mace.pt")
    assert argv[argv.index("--device") + 1] == "cpu"
    mdi = argv[argv.index("-mdi") + 1]
    assert mdi == ("-role ENGINE -name XNN -method TCP -port 8021 -hostname localhost")


def test_get_mdi_engine_command_thread_prefix(ini, monkeypatch):
    monkeypatch.setattr(XnnStep, "thread_count", classmethod(lambda cls: 4))
    executor = SimpleNamespace(name="local")
    argv = XnnStep.get_mdi_engine_command(
        executor, {"root": str(ini)}, method="water_mace", port=1, hostname="h"
    )
    assert argv[:2] == ["OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4"]


def test_get_mdi_engine_command_unknown_model(ini):
    executor = SimpleNamespace(name="local")
    with pytest.raises(ValueError, match="not an available xnn model"):
        XnnStep.get_mdi_engine_command(
            executor, {"root": str(ini)}, method="nope", port=1, hostname="h"
        )


def test_get_mdi_engine_command_local_install(tmp_path, models_dir, monkeypatch):
    monkeypatch.setattr(XnnStep, "thread_count", classmethod(lambda cls: None))
    root = tmp_path / "root2"
    root.mkdir()
    (root / "xnn.ini").write_text(
        "[local]\ninstallation = local\ncode = /usr/local/bin/xnn\n"
        f"device = mps\nmodels = {models_dir}\n"
    )
    executor = SimpleNamespace(name="local")
    argv = XnnStep.get_mdi_engine_command(
        executor, {"root": str(root)}, method="water_mace", port=1, hostname="h"
    )
    assert argv[:2] == ["/usr/local/bin/xnn", "mdi"]
    assert argv[argv.index("--device") + 1] == "mps"


def test_default_ini_bootstraps(tmp_path, monkeypatch):
    """With no xnn.ini, the plug-in's default is written into the root."""
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    executor = SimpleNamespace(name="local")
    config = XnnStep.get_executor_config(executor, {"root": str(tmp_path)})
    assert (tmp_path / "xnn.ini").exists()
    assert config["conda"] == "/opt/conda/bin/conda"
    assert config["conda-environment"] == "seamm-xnn"
    assert "{root}/data/Forcefields/xnn" in config["models"]


def test_thread_count_from_seamm_ini(tmp_path, monkeypatch):
    ini = tmp_path / "seamm.ini"
    monkeypatch.setenv("SEAMM_INI", str(ini))
    ini.write_text("[xnn-step]\nncores = available\n")
    assert XnnStep.thread_count() is None
    ini.write_text("[xnn-step]\nncores = 6\n")
    assert XnnStep.thread_count() == 6
    ini.unlink()
    assert XnnStep.thread_count() is None

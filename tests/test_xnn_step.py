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


def test_personal_and_local_sources(tmp_path, monkeypatch):
    """personal:/local: entries map to the SEAMM data directories, personal shadows
    local, and the reported source uses the Forcefield-step convention."""
    personal = tmp_path / "personal" / "xnn"
    local = tmp_path / "local" / "xnn"
    personal.mkdir(parents=True)
    local.mkdir(parents=True)
    (personal / "mine.pt").write_bytes(b"x")
    (personal / "shared.pt").write_bytes(b"p")
    (local / "shared.pt").write_bytes(b"l")
    (local / "site.pt").write_bytes(b"x")
    monkeypatch.setattr(
        XnnStep,
        "_SOURCE_ROOTS",
        {"personal": tmp_path / "personal", "local": tmp_path / "local"},
    )
    # A root with no data of its own, so local: is just the default installation's
    monkeypatch.setattr(XnnStep, "seamm_root", staticmethod(lambda: tmp_path / "root"))
    config = {"models": "personal:xnn\n    local:xnn\n", "pattern": "*.pt"}
    models, sources = XnnStep.available_models(config, with_sources=True)
    assert sorted(models) == ["mine", "shared", "site"]
    assert models["shared"] == personal / "shared.pt"  # personal wins
    assert sources == {
        "mine": "personal:xnn/mine.pt",
        "shared": "personal:xnn/shared.pt",
        "site": "local:xnn/site.pt",
    }
    options = XnnStep.get_model_chemistry_options.__func__  # noqa: F841
    monkeypatch.setattr(XnnStep, "_local_config", classmethod(lambda cls: config))
    assert (
        XnnStep.get_model_chemistry_options()["mine"]["source"]
        == "personal:xnn/mine.pt"
    )


def test_model_directories_default_ini_lists_personal_then_local():
    import configparser

    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(XnnStep._default_ini_text())
    dirs = XnnStep.model_directories(dict(cfg.items("local")))
    assert [prefix for prefix, _ in dirs] == ["personal:xnn/", "local:xnn/"]
    assert dirs[0][1] == (XnnStep._SOURCE_ROOTS["personal"] / "xnn").expanduser()


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
    assert "personal:xnn" in config["models"] and "local:xnn" in config["models"]


def test_thread_count_from_seamm_ini(tmp_path, monkeypatch):
    ini = tmp_path / "seamm.ini"
    monkeypatch.setenv("SEAMM_INI", str(ini))
    ini.write_text("[xnn-step]\nncores = available\n")
    assert XnnStep.thread_count() is None
    ini.write_text("[xnn-step]\nncores = 6\n")
    assert XnnStep.thread_count() == 6
    ini.unlink()
    assert XnnStep.thread_count() is None


def test_local_source_is_installation_then_default(tmp_path, monkeypatch):
    """local: is <root>/data/Forcefields first, then ~/SEAMM/data/Forcefields."""
    default = tmp_path / "SEAMM" / "data" / "Forcefields"
    root = tmp_path / "SEAMM_DEV"
    own = root / "data" / "Forcefields"
    for d in (default / "xnn", own / "xnn"):
        d.mkdir(parents=True)
    (default / "xnn" / "shared.pt").write_bytes(b"d")
    (default / "xnn" / "both.pt").write_bytes(b"d")
    (own / "xnn" / "dev.pt").write_bytes(b"o")
    (own / "xnn" / "both.pt").write_bytes(b"o")
    monkeypatch.setattr(
        XnnStep,
        "_SOURCE_ROOTS",
        {"personal": tmp_path / "personal", "local": default},
    )
    monkeypatch.setattr(XnnStep, "seamm_root", staticmethod(lambda: root))
    config = {"models": "local:xnn\n", "pattern": "*.pt"}
    assert [d for _, d in XnnStep.model_directories(config)] == [
        own / "xnn",
        default / "xnn",
    ]
    models = XnnStep.available_models(config)
    assert sorted(models) == ["both", "dev", "shared"]
    assert models["both"] == own / "xnn" / "both.pt"  # the installation's wins


# ---------------------------------------------------------------------------
# The training configuration stored in a checkpoint, read without PyTorch
# ---------------------------------------------------------------------------


def _write_checkpoint(path, extra, name="mace", cutoff=6.0, subtracted=None):
    """Write a file laid out like ``torch.save({"model": ..., "cfg": ...})``.

    A zip archive with ``<archive>/data.pkl``: the state dict holds a tensor
    rebuilt by ``torch._utils._rebuild_tensor_v2`` from a persistent storage
    id, and ``cfg`` is a dataclass from a module the reader cannot import --
    as in a real xnn checkpoint. PyTorch itself is not needed: a stand-in
    ``torch._utils`` module is registered only while pickling.
    """
    import collections
    import dataclasses
    import io
    import pickle
    import sys
    import types
    import zipfile

    @dataclasses.dataclass
    class ModelConfig:
        name: str
        cutoff: float
        extra: dict

    @dataclasses.dataclass
    class Config:
        model: ModelConfig
        seed: int = 0
        subtracted_dispersion: object = None

    for cls in (ModelConfig, Config):  # pickled by reference to this module
        cls.__module__ = "xnn.common.config.schema"
        cls.__qualname__ = cls.__name__

    def _rebuild_tensor_v2(*args):
        return None

    class Storage:
        pass

    class Tensor:
        def __reduce__(self):
            return (_rebuild_tensor_v2, (Storage(), 0, (3,), (1,)))

    fake = {
        "torch": types.ModuleType("torch"),
        "torch._utils": types.ModuleType("torch._utils"),
        "xnn": types.ModuleType("xnn"),
        "xnn.common": types.ModuleType("xnn.common"),
        "xnn.common.config": types.ModuleType("xnn.common.config"),
        "xnn.common.config.schema": types.ModuleType("xnn.common.config.schema"),
    }
    fake["torch._utils"]._rebuild_tensor_v2 = _rebuild_tensor_v2
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    fake["xnn.common.config.schema"].ModelConfig = ModelConfig
    fake["xnn.common.config.schema"].Config = Config

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", "FloatStorage", "0", "cpu", 3)
            return None

    saved = {k: sys.modules.get(k) for k in fake}
    sys.modules.update(fake)
    try:
        buffer = io.BytesIO()
        state = collections.OrderedDict([("model.weight", Tensor())])
        cfg = Config(ModelConfig(name, cutoff, extra), subtracted_dispersion=subtracted)
        Pickler(buffer, protocol=2).dump({"model": state, "cfg": cfg})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("best/data.pkl", buffer.getvalue())
        z.writestr("best/byteorder", "little")
        z.writestr("best/data/0", b"\0" * 12)
    return path


D4_B = {
    "name": "d4",
    "cutoff_pair": 12.0,
    "switch_width_pair": 2.0,
    "cutoff_triple": 10.0,
    "tail_correction": True,
}


@pytest.fixture()
def d4_models(tmp_path):
    """Checkpoints for the three dispersion cases plus a double-removed one."""
    d = tmp_path / "d4models"
    d.mkdir()
    les = {"n_channels": 4, "sigma": 1.0, "dl": 1.5}
    _write_checkpoint(d / "plain.pt", {"species": [1, 8], "long_range": les})
    _write_checkpoint(
        d / "route_a.pt",
        {"species": [1, 6, 8], "long_range": les, "dispersion": dict(D4_B)},
    )
    _write_checkpoint(
        d / "route_b.pt",
        {"species": [1, 3, 5, 9], "long_range": les},
        subtracted=dict(D4_B),
    )
    _write_checkpoint(
        d / "twice.pt", {"species": [1, 8], "dispersion": True}, subtracted=True
    )
    return d


def test_read_checkpoint_config_without_torch(d4_models):
    import importlib.util

    from xnn_step.checkpoint import read_checkpoint_config

    cfg = read_checkpoint_config(d4_models / "route_a.pt")
    assert cfg["model"]["name"] == "mace"
    assert cfg["model"]["cutoff"] == 6.0
    assert cfg["model"]["extra"]["species"] == [1, 6, 8]
    assert cfg["model"]["extra"]["dispersion"] == D4_B
    assert cfg["seed"] == 0
    if importlib.util.find_spec("torch") is None:
        import sys

        assert "torch" not in sys.modules  # nothing was imported to read it


def test_read_checkpoint_config_tolerates_other_files(tmp_path):
    from xnn_step.checkpoint import read_checkpoint_config

    (tmp_path / "junk.pt").write_bytes(b"x")
    assert read_checkpoint_config(tmp_path / "junk.pt") is None
    assert read_checkpoint_config(tmp_path / "missing.pt") is None


def test_dispersion_settings():
    from xnn_step.checkpoint import dispersion_settings

    assert dispersion_settings(None) == (None, None)
    assert dispersion_settings({"model": {"extra": {}}}) == (None, None)
    cfg = {"model": {"extra": {"dispersion": True}}}
    assert dispersion_settings(cfg) == ({"name": "d4"}, None)
    cfg = {"model": {"extra": {}}, "subtracted_dispersion": dict(D4_B)}
    assert dispersion_settings(cfg) == (None, D4_B)
    cfg = {"model": {"extra": {}}, "subtracted_dispersion": "D4"}
    assert dispersion_settings(cfg) == (None, {"name": "d4"})
    # the engine ignores a record under model.extra, so the plug-in does too
    cfg = {"model": {"extra": {"subtracted_dispersion": dict(D4_B)}}}
    assert dispersion_settings(cfg) == (None, None)
    cfg = {"model": {"extra": {"dispersion": True}}, "subtracted_dispersion": True}
    with pytest.raises(ValueError, match="removed twice"):
        dispersion_settings(cfg)


def test_model_chemistry_options_describe_the_checkpoint(d4_models, monkeypatch):
    config = {"models": str(d4_models), "pattern": "*.pt"}
    monkeypatch.setattr(XnnStep, "_local_config", classmethod(lambda cls: config))
    options = XnnStep.get_model_chemistry_options()
    # dispersion removed twice: not offered
    assert set(options) == {"plain", "route_a", "route_b"}
    assert options["plain"]["elements"] == "1,8"
    assert options["plain"]["family"] == "mace"
    assert options["plain"]["dispersion"] == ""
    assert options["route_a"]["elements"] == "1,6,8"
    assert options["route_a"]["dispersion"] == "D4, 12 Å + tail in the model"
    assert options["route_b"]["dispersion"] == "D4, 12 Å + tail added by the engine"
    assert "MACE, D4" in options["route_b"]["description"]


def _command(root, method, charge=0):
    return XnnStep.get_mdi_engine_command(
        SimpleNamespace(name="local"),
        {"root": str(root)},
        method=method,
        port=8021,
        hostname="localhost",
        charge=charge,
    )


@pytest.fixture()
def d4_ini(tmp_path, d4_models, monkeypatch):
    monkeypatch.setattr(XnnStep, "thread_count", classmethod(lambda cls: None))
    root = tmp_path / "d4root"
    root.mkdir()
    (root / "xnn.ini").write_text(
        "[local]\ninstallation = local\ncode = xnn\ndevice = cuda\n"
        f"models = {d4_models}\n"
    )
    return root


def test_engine_command_leaves_dispersion_to_the_engine(d4_ini):
    """xnn mdi adds a recorded or built-in term itself; the plug-in passes nothing."""
    for model in ("route_a", "route_b", "plain"):
        assert "--dispersion" not in _command(d4_ini, model)


def test_engine_command_passes_the_total_charge(d4_ini):
    assert "--total-charge" not in _command(d4_ini, "plain", charge=0)
    argv = _command(d4_ini, "plain", charge=-1)
    assert argv[argv.index("--total-charge") + 1] == "-1"
    argv = _command(d4_ini, "route_b", charge=1)
    assert argv[argv.index("--total-charge") + 1] == "1"
    assert argv.index("--total-charge") < argv.index("-mdi")


def test_engine_command_refuses_a_double_removed_model(d4_ini):
    with pytest.raises(ValueError, match="removed twice"):
        _command(d4_ini, "twice")


def test_installer_manages_torch(monkeypatch):
    """The installer hands torch to seamm-manager's driver-aware path, and the
    environment file no longer names torch (seamm_manager#31)."""
    import importlib.resources

    import yaml

    from xnn_step.installer import Installer

    monkeypatch.setattr("sys.argv", ["xnn-step-installer", "show"])
    installer = Installer()
    assert installer.torch_managed is True
    assert "torch" in installer.torch_imports and "xnn" in installer.torch_imports

    data = importlib.resources.files("xnn_step") / "data"
    env = yaml.safe_load((data / "seamm-xnn.yml").read_text())
    pip = [str(r) for d in env["dependencies"] if isinstance(d, dict) for r in d["pip"]]
    assert not any(r == "torch" or r.startswith("torch=") for r in pip)
    assert any(r.startswith("xnns") for r in pip)
    assert "pytorch" not in [str(d).split("=")[0] for d in env["dependencies"]]
    assert "torch-build = auto" in (data / "xnn.ini").read_text()

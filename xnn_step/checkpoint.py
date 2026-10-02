# -*- coding: utf-8 -*-

"""Read the training configuration stored in an xnn checkpoint, without PyTorch.

An xnn checkpoint is what ``Trainer.save`` writes with ``torch.save``:
``{"model": state_dict, "cfg": Config}``. On disk that is a zip archive whose
``<archive>/data.pkl`` is a pickle of the dictionary, with every tensor stored
separately and referenced by a persistent id. The configuration is ordinary
Python data (dataclasses holding numbers, strings, lists and dicts), so it can
be recovered with a restricted unpickler:

* every class the pickle names (``xnn...Config``, ``torch.dtype``, ...) is
  replaced by an inert stand-in that just records its state, so nothing from
  the file is executed -- unlike ``torch.load(weights_only=False)``;
* tensor storages are never read (persistent ids resolve to ``None``).

This lets the SEAMM side, which has no PyTorch, see what a model is: its family,
cutoff and elements, and how dispersion enters it.

Dispersion conventions (see :func:`dispersion_settings`):

``cfg.model.extra["dispersion"]``
    The model was trained with the D3/D4 wrapper, so the checkpoint carries the
    term and the ``xnn mdi`` engine adds it by itself (route A).
``cfg.subtracted_dispersion``
    The model was trained *without* the wrapper, on targets from which this
    dispersion term was subtracted (route B). The record is top-level in the
    configuration, and ``xnn mdi`` (xnns 0.4.0 and later) adds the identical
    term back by itself, so here too the plug-in passes nothing. The value is a
    mapping of the term's options with ``name`` set, a bare name such as
    ``"d4"``, or ``True`` for the D4 defaults.
"""

import functools
import logging
import pickle
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)

#: The ``cfg.model.extra`` key of a dispersion term carried by the model (route A).
IN_MODEL_KEY = "dispersion"
#: The top-level ``cfg`` key recording the dispersion term subtracted from the
#: training targets, which the engine adds back (route B).
SUBTRACTED_KEY = "subtracted_dispersion"


class _Stub:
    """Stand-in for any class named in the pickle: it only records its state."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif (
            isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict)
        ):  # (dict, slots) form
            self.__dict__.update(state[0] or {})
            self.__dict__.update(state[1])


# Callables the restricted unpickler may really construct: plain containers only.
_SAFE = {
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
    ("builtins", "complex"),
    ("copyreg", "_reconstructor"),
}


class _ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return super().find_class(module, name)
        return type(name, (_Stub,), {"__module__": module})

    def persistent_load(self, pid):
        return None  # a tensor storage: not needed, never read


def _plain(obj):
    """Convert the unpickled stand-ins to plain dicts, lists and scalars."""
    if isinstance(obj, _Stub):
        return {k: _plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)


@functools.lru_cache(maxsize=256)
def _read(path, mtime_ns, size):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "data.pkl"]
        if not names:
            raise ValueError("no data.pkl in the archive")
        with archive.open(names[0]) as fd:
            data = _ConfigUnpickler(fd).load()
    if not isinstance(data, dict) or "cfg" not in data:
        raise ValueError("not a trainer checkpoint (no 'cfg')")
    return _plain(data["cfg"])


def read_checkpoint_config(path):
    """The training configuration stored in an xnn checkpoint, as plain data.

    Parameters
    ----------
    path : str or pathlib.Path
        The checkpoint (``.pt``) file.

    Returns
    -------
    dict or None
        The ``Config`` as nested dicts (``cfg["model"]["extra"]`` etc.), or
        ``None`` if the file cannot be read as a trainer checkpoint (the reason
        is logged). Results are cached per file path, size and modification
        time.
    """
    path = Path(path)
    try:
        stat = path.stat()
        return _read(str(path), stat.st_mtime_ns, stat.st_size)
    except Exception as e:
        logger.info(f"Could not read the configuration in {path}: {e}")
        return None


def dispersion_settings(cfg):
    """How dispersion enters a model, from its stored configuration.

    Parameters
    ----------
    cfg : dict or None
        As returned by :func:`read_checkpoint_config`.

    Returns
    -------
    (dict or None, dict or None)
        ``(in_model, subtracted)``: the dispersion term the checkpoint carries,
        and the term subtracted from the training targets. The engine adds
        either by itself. ``True`` is normalized to ``{"name": "d4"}``, the
        PBE0-D4 defaults, and a bare name ``"d3"``/``"d4"`` to ``{"name": ...}``.

    Raises
    ------
    ValueError
        If the configuration has both: the model was then trained with the
        wrapper on dispersion-subtracted targets, which removes the dispersion
        twice, and no run-time setting can correct it.
    """
    cfg = cfg or {}
    extra = (cfg.get("model") or {}).get("extra") or {}

    def normalize(value):
        if not value:
            return None
        if value is True:
            return {"name": "d4"}
        if isinstance(value, str):
            return {"name": value.lower()}
        if isinstance(value, dict):
            return dict(value)
        raise ValueError(f"unrecognized dispersion setting {value!r}")

    in_model = normalize(extra.get(IN_MODEL_KEY))
    subtracted = normalize(cfg.get(SUBTRACTED_KEY))
    if in_model and subtracted:
        raise ValueError(
            f"the model was trained with a dispersion term ({in_model}) on "
            f"targets with dispersion already subtracted ({subtracted}); the "
            "dispersion is removed twice and the model cannot be used as is."
        )
    return in_model, subtracted


def describe_dispersion(spec):
    """A short label for a dispersion mapping, e.g. ``D4`` or ``D4, 12 Å``."""
    if not spec:
        return ""
    label = str(spec.get("name", "d4")).upper()
    if "cutoff_pair" in spec:
        label += f", {spec['cutoff_pair']:g} Å"
    if spec.get("tail_correction"):
        label += " + tail"
    return label

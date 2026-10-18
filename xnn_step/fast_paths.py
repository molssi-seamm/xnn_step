# -*- coding: utf-8 -*-

"""xnn's fast paths: optional packages that make a model much faster on an NVIDIA GPU.

xnn uses them by itself when they are installed, and says nothing when they are not;
without them a large system runs up to twice as slowly on the GPU:

* ``vesin-torch``, fast neighbour lists, and
* ``cuequivariance-torch`` with the operations matching torch's CUDA,
  ``cuequivariance-ops-torch-cu12`` or ``-cu13``, fused tensor products for MACE.

vesin-torch forces float64, which Apple's GPU (mps) does not have, so neither is part
of seamm-xnn.yml: the installer adds them when the environment's torch is a CUDA build.
"""

import json

#: The fast paths, by module, and what they are for the output
FAST_PATHS = {
    "vesin_torch": "the vesin-torch neighbour lists",
    "cuequivariance_torch": "the cuEquivariance tensor products",
}

#: The oldest cuEquivariance with operations for CUDA 13
CUEQUIVARIANCE_MIN = "0.7"

#: Prints the torch build and which fast paths import, without importing torch
PROBE = (
    "import importlib.metadata as md, importlib.util as u, json\n"
    "r = {'modules': {m: u.find_spec(m) is not None for m in %r}}\n"
    "try:\n"
    "    r['torch'] = md.version('torch')\n"
    "except md.PackageNotFoundError:\n"
    "    r['torch'] = None\n"
    "print('FASTPATHS ' + json.dumps(r))\n"
) % (list(FAST_PATHS),)


def parse_probe(output):
    """The probe's dictionary from its output, or None."""
    for line in reversed((output or "").splitlines()):
        if line.startswith("FASTPATHS "):
            try:
                return json.loads(line[10:])
            except json.JSONDecodeError:
                return None
    return None


def cuda_major(torch_version):
    """The CUDA major version of a torch build, e.g. "13" for 2.14.1+cu130; None
    for a CPU build or one that does not say."""
    if not torch_version or "+cu" not in torch_version:
        return None
    digits = torch_version.split("+cu", 1)[1]
    if len(digits) < 3 or not digits.isdigit():
        return None
    return digits[:-1]


def requirements(probe):
    """What to pip install for the missing fast paths, keeping torch as it is.

    Returns
    -------
    ([str], [str])
        The requirements, empty when nothing is to be done, and sentences for what
        cannot be installed.
    """
    notes = []
    torch = (probe or {}).get("torch")
    major = cuda_major(torch)
    if major is None:
        return [], notes
    modules = probe.get("modules", {})
    reqs = []
    if not modules.get("vesin_torch"):
        reqs.append("vesin-torch")
    if not modules.get("cuequivariance_torch"):
        if major in ("12", "13"):
            reqs += [
                f"cuequivariance-torch>={CUEQUIVARIANCE_MIN}",
                f"cuequivariance-ops-torch-cu{major}>={CUEQUIVARIANCE_MIN}",
            ]
        else:
            notes.append(
                f"cuEquivariance has no operations for CUDA {major}, the build of "
                f"torch {torch}, so MACE models run without fused tensor products."
            )
    if reqs:
        reqs.append(f"torch=={torch}")  # pip may not change torch to suit them
    return reqs, notes


def describe(modules):
    """A sentence about the fast paths a GPU run has, and whether it is a warning.

    Returns
    -------
    (str, bool)
    """
    have = [text for module, text in FAST_PATHS.items() if modules.get(module)]
    missing = [text for module, text in FAST_PATHS.items() if not modules.get(module)]
    if not missing:
        return (
            f"On the GPU the engine uses xnn's fast paths: {' and '.join(have)}.",
            False,
        )
    return (
        f"The engine's environment lacks {' and '.join(missing)}, so on the GPU the "
        "model runs without these fast paths of xnn, up to twice as slowly for large "
        "systems. 'xnn-step-installer update' adds them to an environment SEAMM "
        "manages; otherwise install them with pip.",
        True,
    )

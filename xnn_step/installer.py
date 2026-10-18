# -*- coding: utf-8 -*-

"""Installer for the xnn plug-in.

This handles any further installation needed after installing the Python package
`xnn-step`: the `xnn` machine-learned force-field code lives in its own conda
environment (`seamm-xnn`, with PyTorch), which this installer creates from the
bundled `seamm-xnn.yml` and registers in `xnn.ini`.
"""

import importlib
import logging
from pathlib import Path
import subprocess

import seamm_manager

from . import fast_paths

logger = logging.getLogger(__name__)


class Installer(seamm_manager.InstallerBase):
    """Handle further installation needed after installing xnn-step.

    The Python package `xnn-step` should already be installed, using `pip`,
    `conda`, or similar. This plug-in-specific installer then checks for the
    `xnn` command and its location in `xnn.ini`:

    #. If it is already registered in `xnn.ini` there is nothing else to do.
    #. If it can be found using `which` its location is added to `xnn.ini`.
    #. Otherwise it is installed in a separate conda environment, `seamm-xnn`
       by default, using the `seamm-xnn.yml` environment file shipped in the
       plug-in's `data/` directory.

    PyTorch is not in that file. The installer puts it in itself, from the
    PyTorch index that matches the machine's NVIDIA driver (or PyPI on macOS,
    for MPS), leaves a torch that works alone, and checks afterwards that torch
    sees the GPU and that `xnn` and `mdi` import. `torch-build` in `xnn.ini`
    (or `--torch-tag`) forces a build, e.g. on a cluster login node without a
    GPU. See `seamm_manager.torch_support`.
    """

    def __init__(self, logger=logger):
        super().__init__(logger=logger)

        logger.debug("Initializing the xnn installer object.")

        self.environment = "seamm-xnn"
        self.section = "xnn-step"
        self.executables = ["xnn"]
        # PyTorch is installed for this machine's NVIDIA driver, a working
        # torch is left alone, and the environment is checked after install and
        # update: see seamm_manager.torch_support and seamm_manager#31.
        self.torch_managed = True
        self.torch_imports = ("torch", "xnn", "mdi")

        self.resource_path = importlib.resources.files("xnn_step") / "data"

        logger.debug(f"data directory: {self.resource_path}")
        self.environment_file = self.resource_path / "seamm-xnn.yml"

    def _ensure_torch(self, environment, apply_file=True):
        """Bring torch into line with the machine, then add xnn's fast paths for a
        CUDA build of torch (see :mod:`xnn_step.fast_paths`)."""
        ok = super()._ensure_torch(environment, apply_file=apply_file)
        if ok:
            self._ensure_fast_paths(environment)
        return ok

    def _ensure_fast_paths(self, environment):
        """Install the fast paths an environment SEAMM manages is missing.

        Only for a CUDA build of torch, and never in an environment SEAMM did not
        create, for which the installer only says what is missing. torch is pinned
        to the installed version, so that pip cannot change it to suit them.
        """
        probe = fast_paths.parse_probe(
            self.conda.run_python(environment, fast_paths.PROBE)
        )
        if probe is None:
            return
        reqs, notes = fast_paths.requirements(probe)
        for note in notes:
            print(f"    {note}")
        if not reqs:
            if fast_paths.cuda_major(probe.get("torch")) is not None:
                print("    " + fast_paths.describe(probe.get("modules", {}))[0])
            return
        if self._not_ours(environment):
            print("!   " + fast_paths.describe(probe.get("modules", {}))[0])
            return
        print(f"    Installing xnn's fast paths for the GPU: {', '.join(reqs[:-1])}.")
        try:
            from seamm_manager import torch_support

            index_args = torch_support.pip_index_args(self.torch_target())
            self.conda.pip_install(environment, reqs, index_args=index_args)
        except Exception as e:
            print(f"!   Could not install the fast paths: {e}")

    def install(self):
        """Install, then use the GPU by default when the environment can.

        The ``device`` in a newly written ``xnn.ini`` is set to ``cuda`` when the
        environment's torch can use an NVIDIA GPU. An existing ``xnn.ini`` is
        never changed: a ``device = cpu`` there may be deliberate, so the
        installer only says that the GPU could be used.
        """
        path = self.root / self.init_file_name
        existed = path.exists()
        super().install()
        if self.shared_codes or not path.exists():
            return
        try:
            probe = self._probe_torch(self.environment)
        except Exception as e:  # never fail the install over the default device
            logger.debug(f"Could not probe torch in {self.environment}: {e}")
            return
        if not (probe or {}).get("cuda_available"):
            return
        self.exe_config.path = path
        device = self.exe_config.get_values("local").get("device", "cpu")
        if device not in (None, "", "cpu"):
            return
        if existed:
            print(
                f"    The GPU could be used: set 'device = cuda' in {path} to run "
                "the models on it."
            )
            return
        self.exe_config.set_value("local", "device", "cuda")
        self.exe_config.save()
        print(f"    torch can use the GPU, so {path} has 'device = cuda'.")

    def exe_version(self, config):
        """Get the version of the xnn package in its environment.

        Parameters
        ----------
        config : dict
            Configuration data for invoking xnn. Typical keys are `conda` (the
            path to the conda executable) and `conda-environment` (the name or
            full path of the environment in which xnn is installed).

        Returns
        -------
        ("xnn", str)
            The code label and its version, `"unknown"` if it cannot be found.
        """
        environment = config["conda-environment"]
        conda = config["conda"]
        # The code is distributed on PyPI as `xnns` (the name `xnn` is an orphaned
        # project awaiting transfer) while the import name is `xnn`, so look for
        # either distribution rather than assuming the import name.
        script = (
            "import importlib.metadata as m; "
            "d = {p.metadata['Name'].lower(): p.version for p in m.distributions()}; "
            "print(d.get('xnns', d.get('xnn', 'unknown')))"
        )

        if environment[0] == "~":
            environment = str(Path(environment).expanduser())
            command = (
                f"'{conda}' run --live-stream -p '{environment}' python -c \"{script}\""
            )
        elif Path(environment).is_absolute():
            command = (
                f"'{conda}' run --live-stream -p '{environment}' python -c \"{script}\""
            )
        else:
            command = (
                f"'{conda}' run --live-stream -n '{environment}' python -c \"{script}\""
            )

        logger.debug(f"    Running {command}")
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                shell=True,
            )
        except Exception as e:
            logger.debug(f"    Failed to run {command}: {e}")
            version = "unknown"
        else:
            version = result.stdout.strip().splitlines()[-1] if result.stdout else ""
            if result.returncode != 0 or version == "":
                version = "unknown"

        return "xnn", version

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
    """

    def __init__(self, logger=logger):
        super().__init__(logger=logger)

        logger.debug("Initializing the xnn installer object.")

        self.environment = "seamm-xnn"
        self.section = "xnn-step"
        self.executables = ["xnn"]

        self.resource_path = importlib.resources.files("xnn_step") / "data"

        logger.debug(f"data directory: {self.resource_path}")
        self.environment_file = self.resource_path / "seamm-xnn.yml"

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

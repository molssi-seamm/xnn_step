# -*- coding: utf-8 -*-
"""
Control parameters for the xnn step in a SEAMM flowchart
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class XnnParameters(seamm.Parameters):
    """
    The control parameters for the xnn step.

    The xnn plug-in is a *provider* of model chemistries (its MLFF checkpoints)
    for the Model Chemistry step, and the launcher of the `xnn mdi` engine for
    the steps that drive one (Energy, LAMMPS). It has no user-facing step of its
    own, and so no control parameters beyond the standard `results`.
    """

    parameters = {
        "results": {
            "default": {},
            "kind": "dictionary",
            "default_units": None,
            "enumeration": tuple(),
            "format_string": "",
            "description": "results",
            "help_text": "The results to save to variables or in tables.",
        },
    }

    def __init__(self, defaults={}, data=None):
        """Initialize the parameters, by default with the parameters defined above."""

        logger.debug("XnnParameters.__init__")

        super().__init__(defaults={**XnnParameters.parameters, **defaults}, data=data)

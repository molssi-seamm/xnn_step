# -*- coding: utf-8 -*-

"""Non-graphical part of the xnn step in a SEAMM flowchart.

The xnn plug-in provides machine-learned force fields (MLFFs) trained with xnn
as *model chemistries*: the Model Chemistry step lists the checkpoints found in
the directories configured in ``xnn.ini`` (as ``xnn:MLFF@<model>``), and steps
that drive a model chemistry over MDI -- Energy, LAMMPS, Dimer Builder, Normal
Mode Sampling -- launch ``xnn mdi`` through :class:`xnn_step.XnnStep`.

There is therefore no xnn step to place in a flowchart. This node exists only
to satisfy the plug-in interface; running it explains how to use the models.
"""

import logging
from pathlib import Path
import pprint  # noqa: F401

import xnn_step
import seamm
from seamm_util import ureg, Q_  # noqa: F401
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("xnn")


class Xnn(seamm.Node):
    """
    The non-graphical part of the xnn step in a flowchart.

    See Also
    --------
    TkXnn, XnnParameters, XnnStep
    """

    def __init__(self, flowchart=None, title="xnn", extension=None, logger=logger):
        """A step for xnn in a SEAMM flowchart."""
        logger.debug(f"Creating xnn {self}")

        super().__init__(
            flowchart=flowchart,
            title="xnn",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._metadata = xnn_step.metadata
        self.parameters = xnn_step.XnnParameters()

    @property
    def version(self):
        """The semantic version of this module."""
        return xnn_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return xnn_step.__git_revision__

    def description_text(self, P=None):
        """Create the text description of what this step will do."""
        text = (
            "The xnn plug-in provides machine-learned force fields as model "
            "chemistries; it is not run as a step. Choose an 'xnn' model in a "
            "Model Chemistry step and follow it with e.g. an Energy step."
        )
        return self.header + "\n" + __(text, indent=4 * " ").__str__()

    def run(self):
        """Run the xnn step -- which only explains how the plug-in is used."""
        next_node = super().run(printer)
        printer.important(__(self.description_text(), indent=self.indent))

        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)

        models = xnn_step.XnnStep.available_models()
        if models:
            text = "The xnn models available on this machine are:"
            printer.important(__(text, indent=4 * " "))
            for name, path in models.items():
                printer.important(f"        xnn:MLFF@{name}    ({path})")
        else:
            printer.important(
                __(
                    "No xnn models were found. Put checkpoint files in the "
                    "'models' directories listed in xnn.ini.",
                    indent=4 * " ",
                )
            )
        printer.important("")

        return next_node

    def analyze(self, indent="", **kwargs):
        """Nothing to analyze: this step produces no results."""
        pass

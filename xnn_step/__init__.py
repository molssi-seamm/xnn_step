# -*- coding: utf-8 -*-

"""
xnn_step
A SEAMM plug-in for xnn
"""

# Bring up the classes so that they appear to be directly in
# the xnn_step package.

from .xnn import Xnn  # noqa: F401, E501
from .xnn_parameters import XnnParameters  # noqa: F401, E501
from .xnn_step import XnnStep  # noqa: F401, E501
from .tk_xnn import TkXnn  # noqa: F401, E501

from .metadata import metadata  # noqa: F401

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

# -*- coding: utf-8 -*-

"""Handle the installation of the xnn step."""

from .installer import Installer


def run():
    """Handle the extra installation needed.

    * Find and/or install the `xnn` command, by default in the conda
      environment `seamm-xnn`.
    * Add or update information in the SEAMM xnn.ini file.
    """

    # Create an installer object
    installer = Installer()
    installer.run()


if __name__ == "__main__":
    run()

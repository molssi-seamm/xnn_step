=================
SEAMM xnn Plug-in
=================

.. image:: https://img.shields.io/github/issues-pr-raw/molssi-seamm/xnn_step
   :target: https://github.com/molssi-seamm/xnn_step/pulls
   :alt: GitHub pull requests

.. image:: https://github.com/molssi-seamm/xnn_step/workflows/CI/badge.svg
   :target: https://github.com/molssi-seamm/xnn_step/actions
   :alt: Build Status

.. image:: https://codecov.io/gh/molssi-seamm/xnn_step/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/molssi-seamm/xnn_step
   :alt: Code Coverage

.. image:: https://github.com/molssi-seamm/xnn_step/workflows/CodeQL/badge.svg
   :target: https://github.com/molssi-seamm/xnn_step/security/code-scanning
   :alt: Code Quality

.. image:: https://github.com/molssi-seamm/xnn_step/workflows/Release/badge.svg
   :target: https://molssi-seamm.github.io/xnn_step/index.html
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/xnn_step.svg
   :target: https://pypi.python.org/pypi/xnn_step
   :alt: PyPi VERSION

A SEAMM plug-in providing machine-learned force fields (MLFFs) trained with `xnn`_ as
model chemistries, run as MDI engines.

* Free software: BSD-3-Clause
* Documentation: https://molssi-seamm.github.io/xnn_step/index.html
* Code: https://github.com/molssi-seamm/xnn_step

Features
--------

* Advertises every xnn checkpoint found in the directories listed in ``xnn.ini`` to
  the Model Chemistry step as ``xnn:MLFF@<model>``.
* Launches ``xnn mdi`` -- the checkpoint served as a resident MDI engine -- for any step
  that drives a model chemistry over MDI: Energy (single points for many structures),
  LAMMPS (MD with the MLFF as the QM engine), Dimer Builder, Normal Mode Sampling.
* Molecular and periodic systems (energy, forces and stress); runs on the CPU or a GPU
  (``device`` in ``xnn.ini``).
* There is no xnn step to place in a flowchart: the models are used through
  Model Chemistry + Energy (or LAMMPS).

.. _xnn: https://github.com/molssi-ai/xnn

Acknowledgements
----------------

This package was created with the `molssi-seamm/cookiecutter-seamm-plugin`_ tool, which
is based on the excellent Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`molssi-seamm/cookiecutter-seamm-plugin`: https://github.com/molssi-seamm/cookiecutter-seamm-plugin

Developed by the Molecular Sciences Software Institute (MolSSI_),
which receives funding from the `National Science Foundation`_ under
award CHE-2136142.

.. _MolSSI: https://molssi.org
.. _`National Science Foundation`: https://www.nsf.gov

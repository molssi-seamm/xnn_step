***************
Getting Started
***************

Installation
============
The xnn plug-in is probably already installed in your SEAMM environment, but if not or
if you wish to check, follow the directions for the `SEAMM Installer`_. In the SEAMM
conda environment::

  seamm-installer install --update xnn-step

The plug-in needs the `xnn`_ code itself (published on PyPI as ``xnns``), with PyTorch, in
its own conda environment
(``seamm-xnn`` by default). ``xnn-step-installer`` creates it from the bundled
``seamm-xnn.yml`` and records it in ``~/SEAMM/xnn.ini``::

  xnn-step-installer install

The installer chooses the PyTorch build for the machine. PyTorch wheels bundle their
CUDA runtime, so the NVIDIA *driver* decides which build runs, and PyPI's default wheel
-- which bundles the newest runtime -- silently falls back to the CPU on an older driver.
The installer reads the driver with ``nvidia-smi``, installs torch from the matching
PyTorch index (PyPI's own wheel on macOS, which supports Apple's ``mps``), and then
checks that torch sees the GPU and that ``xnn`` imports; it prints what it found. Using a
GPU then needs no more than ``device = cuda`` in ``xnn.ini``.

On a machine where the driver cannot be read -- a cluster whose login node has no GPU --
the installer stops and asks you to decide. Set ``torch-build`` in ``xnn.ini`` (``cu128``
for a current driver, ``cpu`` for a machine without a GPU) or run::

  xnn-step-installer install --torch-tag cu128

A torch that already works is never replaced. One that cannot use the GPU (for instance
installed from PyPI before this) is reported and left alone, since it may be deliberate;
``xnn-step-installer update --torch-tag cu128`` replaces it.

``pymdi``, which provides the MDI library, comes from conda-forge: only that build links
MDI against MPI, which the ``-method MPI`` launch used for LAMMPS dynamics needs.

.. note::
   ``conda-environment`` in ``xnn.ini`` may name an environment you built yourself, for
   instance one shared with LAMMPS. The installer recognises an environment it did not
   create (from conda's own history) and leaves it alone; to have it add xnn to such an
   environment anyway, run its ``update --torch-tag <tag>`` deliberately. The file's
   pip part is applied without upgrading what is already present, on torch's index.

.. _SEAMM Installer: https://molssi-seamm.github.io/installation/index.html
.. _xnn: https://github.com/molssi-ai/xnn

Adding models
=============
Put the trained checkpoints (``best.pt`` files written by the xnn trainer) in
``~/.seamm.d/data/Forcefields/xnn`` (your own models) or
``~/SEAMM/data/Forcefields/xnn`` (models shared by everyone on the machine). Those two
directories are searched by default, and a personal model shadows a machine-wide one of
the same name. Other directories can be listed under ``models`` in ``xnn.ini``, one per
line. Each file is offered to the Model Chemistry step by its file stem, so
``~/.seamm.d/data/Forcefields/xnn/water_mace_les.pt`` becomes the model chemistry
``xnn:MLFF@water_mace_les``.

Using a model
=============
In the flowchart editor add a *Model Chemistry* step, choose the type ``MLFF`` and the
model, then follow it with an *Energy* step (to evaluate structures) or a *LAMMPS* step
(to run dynamics with the MLFF as the QM engine). For example, to label an ensemble::

    Parameters -> Model Chemistry (xnn:MLFF@water_mace_les) -> from SMILES
               -> Normal Mode Sampling -> Energy (configurations: all)
               -> Write Structure (Results.extxyz)

That should be enough to get started. For more detail about the functionality in this
plug-in, see the :ref:`User Guide <user-guide>`.

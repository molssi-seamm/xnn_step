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

PyTorch is installed with pip, whose wheels support CUDA on Linux and Apple's ``mps``
on macOS, so using a GPU normally needs no more than ``device = cuda`` in ``xnn.ini``.
To pin a particular CUDA version, install the matching build by hand afterwards,
following https://pytorch.org::

  conda run -n seamm-xnn pip install torch \
      --index-url https://download.pytorch.org/whl/cu126

``pymdi``, which provides the MDI library, comes from conda-forge: only that build links
MDI against MPI, which the ``-method MPI`` launch used for LAMMPS dynamics needs.

.. warning::
   ``conda-environment`` in ``xnn.ini`` may name an environment you built yourself, for
   instance one shared with LAMMPS. ``xnn-step-installer update`` then applies
   ``seamm-xnn.yml`` to *that* environment, so anything in it that the file also names
   may be replaced.

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

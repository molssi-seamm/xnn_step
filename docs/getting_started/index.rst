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

For a GPU, install the matching PyTorch build into that environment afterwards
(see https://pytorch.org) and set ``device = cuda`` in ``xnn.ini``.

.. _SEAMM Installer: https://molssi-seamm.github.io/installation/index.html
.. _xnn: https://github.com/molssi-ai/xnn

Adding models
=============
Put the trained checkpoints (``best.pt`` files written by the xnn trainer) in the
directory ``~/SEAMM/data/Forcefields/xnn`` -- or list other directories under ``models``
in ``xnn.ini``, one per line. Each file is offered to the Model Chemistry step by its file
stem, so ``~/SEAMM/data/Forcefields/xnn/water_mace_les.pt`` becomes the model chemistry
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

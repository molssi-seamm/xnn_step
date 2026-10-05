=======
History
=======
2026.10.5 -- PyTorch installed for the machine's NVIDIA driver, and checked
    * ``xnn-step-installer`` no longer takes whatever torch PyPI serves, which on a
      machine whose driver is older than the wheel's bundled CUDA runtime imports and
      silently runs on the CPU. It reads the driver with ``nvidia-smi``, installs torch
      from the matching PyTorch index (PyPI's wheel on macOS, for mps), applies the
      rest of the environment on that index, leaves a torch that works alone, and
      checks afterwards that torch sees the GPU and that ``xnn`` and ``mdi`` import.
      ``torch-build`` in ``xnn.ini`` or ``--torch-tag`` forces a build, which a
      cluster login node without a GPU needs; a torch that cannot use the GPU is
      replaced only when asked that way. ``torch`` is no longer in
      ``seamm-xnn.yml``. (seamm_manager#31)
    * Requires seamm-manager 2026.10.5.1.

2026.10.2 -- Show what each model is, and pass the charge for D4 models
    * The Model Chemistry step now describes each xnn model: its family (e.g. MACE), the
      elements it was trained on, and how dispersion enters it, e.g. "D4, 12 Å + tail
      added by the engine". The plug-in reads this from the training configuration
      stored in the checkpoint, without PyTorch and without executing anything in the
      file.
    * Models trained on dispersion-subtracted labels, which record the subtracted term
      as ``subtracted_dispersion`` in their training configuration, are recognized; the
      ``xnn mdi`` engine adds the term back by itself. A model that records the term
      and also carries a dispersion wrapper would count the dispersion twice. It is no
      longer offered, and asking for it by name is an error.
    * For a charged configuration the total charge is now passed to the engine as
      ``--total-charge``. D4 dispersion needs it for its EEQ partial charges.

2026.9.28 -- Require xnns 0.4.0, which loads the existing checkpoints
    * The environment now requires ``xnns>=0.4.0`` instead of ``xnns<0.2``. xnns 0.4.0
      loads the checkpoints that 0.2.1 and 0.3.0 could not, and its engine supports
      ``--eeq-reuse``. Updating the plug-in moves an existing environment to it, including
      one rolled back to 0.1.0 by hand; a torch built for the machine's CUDA driver is
      kept, since xnns needs only torch 2.0 or later.

2026.9.27.1 -- Local models can belong to the installation
    * ``local:`` model directories in ``xnn.ini`` meant ``~/SEAMM/data/Forcefields``.
      They now mean the ``data/Forcefields`` directory of the SEAMM installation in use,
      then ``~/SEAMM/data/Forcefields``; a model in the installation's own directory
      takes precedence. A second installation such as ``~/SEAMM_DEV`` therefore sees
      the default installation's models and can add its own.
    * The SEAMM root now comes from ``seamm_util.current_root()``, so an installation's
      ``xnn.ini`` is found without ``--root``. Requires seamm-util 2026.9.27.1.

2026.9.27 -- Bugfix: pin xnns below 0.2 so existing checkpoints load
    * The environment now requires ``xnns<0.2``. The 0.2.1 and 0.3.0 releases on PyPI
      added ``scale_shift`` buffers to the MACE model with no defaults for older state
      dicts, so every existing checkpoint failed to load in the MDI engine ("Missing
      key(s) in state_dict: model.model.scale_shift.scale/shift") and a LAMMPS run then
      hung waiting for it. A fresh installation was getting 0.3.0.
    * Updating does not downgrade an environment that already has a newer ``xnns``;
      run ``pip install xnns==0.1.0`` in it by hand. The pin will move forward once an
      xnns release loads the older checkpoints.

2026.9.26 -- Internal: depend on seamm-manager rather than seamm-installer
    * The plug-in's installer now builds on ``seamm-manager``, which replaces
      ``seamm-installer`` for managing SEAMM installations. Nothing changes for users;
      this only lets the two packages stop being installed side by side.

2026.9.19.1 (2026-09-19)
------------------------

* Bugfix: the ``seamm-xnn`` environment file no longer installs PyTorch with conda. On
  Linux conda-forge resolves ``pytorch`` to a CPU-only build, so updating an environment
  that held a pip CUDA build of PyTorch silently lost the GPU and broke every compiled
  extension built against it, such as ``vesin-torch``. PyTorch now comes from pip, which
  leaves a suitable existing installation alone.
* Bugfix: ``pymdi`` now comes from conda-forge rather than pip. Only that build links the
  MDI library against MPI, which the ``-method MPI`` launch used for LAMMPS dynamics
  needs; with the PyPI build the engine stopped at "Error in MDI_Init: Failed to
  initialize MPI".
* Bugfix: ``xnn-step-installer`` reported the xnn version as "unknown" because it looked
  for a distribution named ``xnn``, though the code is published as ``xnns``.
* Documented the two default model directories and their ``personal:``/``local:``
  prefixes, the MPI launch path, and the effect of pointing ``xnn.ini`` at an
  environment shared with another code.


2026.9.19 (2026-09-19)
----------------------

* Model directories in ``xnn.ini`` may be given as ``personal:<subdir>``
  (``~/.seamm.d/data/Forcefields/<subdir>``) or ``local:<subdir>``
  (``~/SEAMM/data/Forcefields/<subdir>``), the same convention as the Forcefield
  step; the default is ``personal:xnn`` then ``local:xnn``, a personal model
  shadowing a machine one of the same name. Each model's source (e.g.
  ``personal:xnn/water.pt``) is reported to the Model Chemistry step.
* Installs xnn from PyPI as ``xnns``; documents the Apple mps constraints.


2026.9.18 (2026-09-18)
----------------------

* Initial release. Provides xnn machine-learned force fields as ``xnn:MLFF@<model>``
  model chemistries (checkpoints discovered from the directories in ``xnn.ini``) and
  launches ``xnn mdi`` as the MDI engine for the Energy, LAMMPS and other MDI-driving
  steps. Includes ``xnn-step-installer`` for the ``seamm-xnn`` conda environment.
* Plug-in created using the SEAMM plug-in cookiecutter.

=======
History
=======
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

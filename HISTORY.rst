=======
History
=======

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

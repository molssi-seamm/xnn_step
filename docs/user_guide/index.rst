.. _user-guide:

**********
User Guide
**********

The xnn plug-in is a *provider* rather than a step. It connects the machine-learned
force fields trained with `xnn`_ to SEAMM's Model Chemistry mechanism, and to the steps
that drive a model chemistry over the MolSSI Driver Interface (MDI).

.. _xnn: https://github.com/molssi-ai/xnn

xnn.ini
=======
``~/SEAMM/xnn.ini`` (created with defaults the first time it is needed) has a section
per executor, normally just ``[local]``:

``installation``
    ``conda`` (the default) or ``local``.
``conda``, ``conda-environment``
    The conda executable and the environment holding xnn and PyTorch
    (``seamm-xnn``). A path (absolute or ``~/...``) is accepted for the environment.
``code``
    The command to run xnn, ``xnn`` by default; for a ``local`` installation a full
    path may be given.
``device``
    The PyTorch device the engine evaluates on: ``cpu``, ``cuda``, ``cuda:0``, ``mps``.
    On Apple silicon ``mps`` works only for a float32 model and only without the
    optional ``vesin-torch`` neighbor-list package, which insists on float64 (Apple's
    GPU has none); for small molecules the CPU is as fast or faster anyway.
``models``
    The directories searched for checkpoints, one per line. An entry may be
    ``personal:<subdir>`` (``~/.seamm.d/data/Forcefields/<subdir>``),
    ``local:<subdir>`` (``~/SEAMM/data/Forcefields/<subdir>``), or a plain path in which
    ``{root}`` expands to the SEAMM root (``~/SEAMM`` unless ``--root`` was given). The
    default is ``personal:xnn`` then ``local:xnn``, a personal model shadowing a
    machine-wide one of the same name.
``pattern``
    The glob for checkpoint files, ``*.pt`` by default.

The ``[xnn-step]`` section of ``seamm.ini`` has one option, ``ncores``: a cap on the CPU
threads the engine may use (``OMP_NUM_THREADS``), or ``available`` for PyTorch's default.

Model chemistries
=================
Each checkpoint is advertised as ``xnn:MLFF@<name>``. A checkpoint with a name of its
own is called by its file stem: ``water_mace.pt`` is ``xnn:MLFF@water_mace``. Training
writes generic names -- ``runs/<...>/<run>/best.pt`` and ``last.pt`` -- so a checkpoint
named ``best``, ``last``, ``model``, ``checkpoint`` or ``final`` is called after its
directory instead, with the directory above too when its own is only an index (``m1``,
``fold2``, ``seed0``), the stem unless it is ``best``, and the first six hex digits of
the SHA-256 of the file:

* ``.../x013_m1/best.pt`` is ``xnn:MLFF@x013_m1-3f9a2c``;
* ``.../x013_m1/last.pt`` is ``xnn:MLFF@x013_m1-last-7b1e09``;
* ``.../committee_w1/m1/best.pt`` is ``xnn:MLFF@committee_w1-m1-0c44d5``.

The hash makes the name that of these weights, so results from a later or retrained
checkpoint in the same place are not labelled as the same model. To choose a name,
give the checkpoint one, e.g. with a symlink. In every name the grammar's reserved
characters (``: @ / |``) are replaced by ``-``.

The models are all flagged MDI-capable and periodic-capable: the engine builds the
atomic graph with the cell when one is sent and returns the stress.

The MDI engine
==============
A driving step asks the plug-in for the launch command, which runs::

    conda run --live-stream -n seamm-xnn xnn mdi --ckpt <checkpoint> --device <device> \
        -mdi "-role ENGINE -name XNN -method TCP -port <port> -hostname <host>"

The engine loads the model once and then answers ``<ENERGY``, ``<FORCES`` and
``<STRESS`` for every geometry the driver sends (``>COORDS``, and ``>CELL`` for
periodic systems), converting between MDI's atomic units and the model's eV/Å at the
boundary.

Steps that launch the engine and the driver together under ``mpirun``, such as LAMMPS
running dynamics on a GPU, use ``-method MPI`` in place of a TCP port. That path needs
the MDI library itself to be linked against MPI, which is why the environment takes
``pymdi`` from conda-forge rather than PyPI. With the PyPI build the engine stops at
``Error in MDI_Init: Failed to initialize MPI``.

For this the plug-in gives the engine's command without ``conda run``: the engine runs
in the driver's environment, since the two must share MPI and the MDI library, so
LAMMPS's environment must have xnn installed. LAMMPS asks for it when a Model Chemistry
step selects an ``xnn:MLFF`` model, and launches the pair itself (see the LAMMPS step's
documentation)::

    xnn mdi --ckpt <checkpoint> --device cuda:0 <engine options> \
        -mdi "-role ENGINE -name XNN -method MPI"

Checkpoints by path
===================
A Model Chemistry step can name a checkpoint by its path, or with a variable, rather
than choosing one of the models found in the directories above -- e.g. one written
during training. It is offered as ``xnn:MLFF@<file stem>`` and checked as the others
are. The Model Chemistry step's **Engine options** are added to the engine's command for
that run, e.g. ``--eeq-reuse`` or ``--dispersion "{regime: large}"``.

A model whose checkpoint was pickled by an older ``xnns`` release fails to load with
``No module named 'xnn'``; re-save it with the current xnn.

The environment requires ``xnns>=0.4.0``. The 0.2.1 and 0.3.0 releases added
``scale_shift`` buffers to the MACE model without defaults for older state dicts, so every
checkpoint trained before them failed in the engine with ``Missing key(s) in state_dict:
"model.model.scale_shift.scale", "model.model.scale_shift.shift"`` (and a LAMMPS driver
then waited forever for the engine); 0.4.0 loads them again. Updating the plug-in moves
an existing environment to 0.4.0 or later, including one rolled back to 0.1.0 by hand.
xnns needs only ``torch>=2.0``, so the torch the installer chose for the machine's
driver (see :doc:`../getting_started/index`, ``torch-build``) is left as it is.

Committees
==========
A committee -- a directory of members ``m0/best.pt``, ``m1/best.pt``, ..., each trained
on a different part of the data, as the Active Learning step's Train Committee writes
one -- is offered as a single model. In a model directory it is called by the
directory's name (``water_r2/`` is ``xnn:MLFF@water_r2``); given by path, a directory
with a generic name such as ``committee`` is called after its parent with a short hash
of the members. A ``committee.json`` in the directory, e.g. ``{"members": ["m0", "m1",
"m3"]}``, chooses the members; otherwise all are used, and a committee needs at least
two. Every member must handle the dispersion the same way.

A committee runs as the committee-mean engine of the `active-learning-step`_ package,
which must be installed with SEAMM, in the same MLFF environment and the same way as a
single model (``conda run`` and TCP, or as an MPI pair)::

    env PYTHONPATH=<copy of the package> python -m active_learning_step.cli engine \
        --members m0/best.pt m1/best.pt ... --device <device> \
        --spreads committee_spreads.csv --every 10 <engine options> -mdi "..."

The forces, energy and stress are the members' mean, and every ten steps the spreads
of the members -- of the energy, the forces and the pressure -- are appended to
``committee_spreads.csv`` in the step's directory, which the step keeps (the model
chemistry lists the files its engine writes as ``engine_files``). Nothing is installed in the MLFF
environment: a copy of the package's code, made once per version in
``~/.seamm.d/active_learning/code/``, is put on its ``PYTHONPATH``. The Model Chemistry
step's **Engine options** go to this engine, e.g. ``--every 20``, ``--flag
rho_forces=0.2`` (write the frames whose spread reaches it), ``--dtype float32`` or
``--eeq-reuse``.

.. _active-learning-step: https://molssi-seamm.github.io/active_learning_step/

Dispersion and charge
=====================
The plug-in reads the training configuration stored in each checkpoint. It does this
without PyTorch: the configuration is recovered from the checkpoint's pickle with every
class replaced by an inert stand-in, so nothing in the file is executed. From it the
Model Chemistry step is told the model family, the elements the model was trained on,
and how dispersion enters the model. Dispersion can enter in one of two ways, recorded in
the training configuration:

``model.extra["dispersion"]``
    The model was trained with xnn's D3/D4 wrapper, so the checkpoint carries the term.
``subtracted_dispersion`` (top level)
    The model was trained without the wrapper, on energies and forces from which this
    dispersion term was subtracted. ``xnn mdi`` adds the identical term back, using the
    recorded settings, for example ``{name: d4, cutoff_pair: 12.0, switch_width_pair:
    2.0, cutoff_triple: 8.0, tail_correction: true}``.

In both cases the engine adds the dispersion by itself and the plug-in passes nothing;
the Model Chemistry step shows which applies, e.g. "D4, 12 Å + tail added by the engine".
A checkpoint with both was trained with the wrapper on dispersion-subtracted targets,
which removes the dispersion twice. It is not offered as a model chemistry, and asking
for it by name is an error.

The D4 dispersion depends on the total charge of the system through its EEQ partial
charges. For a charged configuration the plug-in passes the charge to the engine as
``--total-charge``.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

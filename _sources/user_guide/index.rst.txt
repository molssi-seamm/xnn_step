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
Each checkpoint is advertised as ``xnn:MLFF@<name>``, where ``<name>`` is the file stem
with any of the grammar's reserved characters (``: @ / |``) replaced by ``-``. The
models are all flagged MDI-capable and periodic-capable: the engine builds the atomic
graph with the cell when one is sent and returns the stress.

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

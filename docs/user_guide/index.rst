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

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

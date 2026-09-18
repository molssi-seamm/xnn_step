# -*- coding: utf-8 -*-

"""Helper class needed for the stevedore integration, plus the Model Chemistry /
MDI engine contract for xnn machine-learned force fields (MLFFs).

The xnn plug-in is a *provider*: it advertises the trained xnn checkpoints found
in the configured model directories to the Model Chemistry step (as
``xnn:MLFF@<model>``), and builds the command that launches ``xnn mdi`` -- the
checkpoint served as a resident MDI engine -- for any step that drives a model
chemistry over MDI (Energy, LAMMPS, Dimer Builder, Normal Mode Sampling, ...).
"""

import configparser
import logging
import os
from pathlib import Path
import shutil

import xnn_step

logger = logging.getLogger(__name__)

# Characters reserved by the model-chemistry grammar; a model's name must not
# contain them, so they are replaced in the file stem.
_RESERVED = ":@/|"


class XnnStep(object):
    """Helper class needed for the stevedore integration.

    This must provide a `description()` method that returns a dict containing a
    description of this node, and `create_node()` and `create_tk_node()` methods
    for creating the graphical and non-graphical nodes.

    The dictionary for the description is the class variable just below these
    comments. The felds are as follows:

        my_description : {str, str}
            A human-readable description of this step. It can be
            several lines long, and needs to be clear to non-expert users.
            It contains the following keys: description, group, name.

        my_description["description"] : tuple
            A description of the xnn step. It must be
            clear to non-experts.

        my_description["group"] : str
            Which group in the menus to put this step. If the group does
            not exist it will be created. Common groups are "Building",
            "Control", "Custom", "Data", and "Simulations".

        my_description["name"] : str
            The name of this step, to be displayed in the menus.
    """

    my_description = {
        "description": (
            "Machine-learned force fields (MLFFs) trained with xnn, provided as "
            "model chemistries and run as MDI engines."
        ),
        "group": "Simulations",
        "name": "xnn",
    }

    # ----------------------------------------------------------------------
    # Model-chemistry / MDI engine contract (mirrors MOPACStep / XTBStep).
    #
    # xnn models are trained checkpoints (`best.pt`), not named methods, so
    # the options are discovered by scanning the directories listed in xnn.ini.
    # Every model is MDI-capable (that is the only way xnn is run here) and
    # handles periodic cells: the engine builds the graph with the cell and
    # returns the stress.
    # ----------------------------------------------------------------------

    @classmethod
    def get_model_chemistry_options(cls, periodic_only=False, mdi_only=False):
        """Return the model chemistries (level specs) the xnn models provide.

        Advertises bare ``xnn:MLFF@<model>`` level specs, one per checkpoint
        found in the model directories configured in ``xnn.ini``, keyed by the
        model name (the checkpoint's file stem, with any grammar-reserved
        characters replaced by ``-``).

        Parameters
        ----------
        periodic_only : bool
            Only return models validated for periodic systems (all of them).
        mdi_only : bool
            Only return models launchable via MDI (all of them).

        Returns
        -------
        dict
            Keyed by model name. Each entry carries ``model_chemistry``,
            ``type`` (``"MLFF"``), ``description``, ``periodic_native``,
            ``periodic_mdi``, ``elements`` (unknown: ``""``), ``mdi_capable``,
            ``mdi_method_arg`` (the model name) and ``path`` (the checkpoint).
        """
        options = {}
        for name, path in cls.available_models().items():
            options[name] = {
                "model_chemistry": f"xnn:MLFF@{name}",
                "type": "MLFF",
                "description": f"xnn machine-learned force field {path.name}",
                "periodic_native": True,
                "periodic_mdi": True,
                "elements": "",
                "mdi_capable": True,
                "mdi_method_arg": name,
                "path": str(path),
            }
        return options

    @classmethod
    def available_models(cls, config=None):
        """The xnn checkpoints found in the configured model directories.

        Parameters
        ----------
        config : dict, optional
            An ``xnn.ini`` section (see :meth:`get_executor_config`). Defaults to
            the ``[local]`` section of the user's ``xnn.ini``, or the plug-in's
            default ini if the user has none.

        Returns
        -------
        dict[str, Path]
            Model name -> checkpoint path, sorted by name. Two files with the same
            stem in different directories keep the first found (a warning is
            logged for the other).
        """
        if config is None:
            config = cls._local_config()

        root = cls.seamm_root()
        pattern = config.get("pattern", "*.pt").strip() or "*.pt"
        models = {}
        for line in config.get("models", "").splitlines():
            directory = line.strip()
            if directory == "" or directory.startswith("#"):
                continue
            directory = Path(directory.replace("{root}", str(root))).expanduser()
            if not directory.is_dir():
                logger.debug(f"xnn model directory {directory} does not exist")
                continue
            for path in sorted(directory.glob(pattern)):
                if not path.is_file():
                    continue
                name = cls.model_name(path)
                if name in models:
                    logger.warning(
                        f"xnn model '{name}' found twice; keeping {models[name]} and "
                        f"ignoring {path}."
                    )
                    continue
                models[name] = path
        return dict(sorted(models.items()))

    @staticmethod
    def model_name(path):
        """The model-chemistry method name for a checkpoint: its file stem with
        the grammar-reserved characters replaced by ``-``."""
        name = Path(path).stem
        for c in _RESERVED:
            name = name.replace(c, "-")
        return name

    @staticmethod
    def seamm_root():
        """The SEAMM root directory (``--root``), as a Path.

        Uses the parsed SEAMM options when available (inside a running flowchart
        or the editor), else ``~/SEAMM``.
        """
        root = None
        try:
            from seamm_util import getParser

            root = getParser().get_options("SEAMM").get("root", None)
        except Exception:
            root = None
        if root is None or root == "":
            root = os.environ.get("SEAMM_ROOT", "~/SEAMM")
        return Path(root).expanduser()

    @classmethod
    def _ini_path(cls, root=None):
        if root is None:
            root = cls.seamm_root()
        return Path(root).expanduser() / "xnn.ini"

    @classmethod
    def _default_ini_text(cls):
        import importlib.resources

        resources = importlib.resources.files("xnn_step") / "data"
        return (resources / "xnn.ini").read_text()

    @classmethod
    def _read_ini(cls, ini_path):
        """Read xnn.ini, bootstrapping it from the plug-in default if missing."""
        if not ini_path.exists():
            boot = configparser.ConfigParser(interpolation=None)
            boot.read_string(cls._default_ini_text())
            if "local" not in boot:
                boot.add_section("local")
            boot["local"]["installation"] = "conda"
            conda = os.environ.get("CONDA_EXE", shutil.which("conda") or "conda")
            boot["local"]["conda"] = conda
            boot["local"].setdefault("conda-environment", "seamm-xnn")
            try:
                ini_path.parent.mkdir(parents=True, exist_ok=True)
                with ini_path.open("w") as fd:
                    boot.write(fd)
            except OSError as e:
                logger.warning(f"Could not write a default {ini_path}: {e}")
            return boot
        config = configparser.ConfigParser(interpolation=None)
        config.read(ini_path)
        return config

    @classmethod
    def _local_config(cls):
        """The ``[local]`` section of the user's xnn.ini, as a dict."""
        try:
            config = cls._read_ini(cls._ini_path())
        except Exception as e:
            logger.warning(f"Could not read xnn.ini: {e}")
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(cls._default_ini_text())
        section = "local" if "local" in config else config.sections()[0]
        return dict(config.items(section))

    @classmethod
    def get_executor_config(cls, executor, seamm_options):
        """Return how to launch the xnn MDI engine on this machine.

        Reads the per-plug-in ``xnn.ini`` for the current executor type, so the
        engine runs in the conda environment holding xnn and PyTorch
        (``seamm-xnn`` by default).

        Parameters
        ----------
        executor : seamm.ExecutorBase
            The flowchart executor (``self.flowchart.executor`` in the driver);
            ``executor.name`` selects the ini section.
        seamm_options : dict
            The global SEAMM options (``self.global_options``);
            ``seamm_options["root"]`` locates the ini.

        Returns
        -------
        dict
            The ini section for the current executor, plus ``version`` (the
            plug-in version).
        """
        executor_type = executor.name
        root = Path(seamm_options["root"]).expanduser()
        ini_path = cls._ini_path(root)

        full_config = cls._read_ini(ini_path)

        # Last-ditch: fall back to an xnn executable on $PATH (a local install).
        if executor_type not in full_config:
            path = shutil.which("xnn")
            if path is None:
                raise RuntimeError(
                    f"No section for '{executor_type}' in the xnn ini file "
                    f"({ini_path}), nor in the defaults, nor on $PATH."
                )
            full_config.add_section(executor_type)
            full_config.set(executor_type, "installation", "local")
            full_config.set(executor_type, "code", str(path))
            with ini_path.open("w") as fd:
                full_config.write(fd)

        config = dict(full_config.items(executor_type))
        config["version"] = xnn_step.__version__
        return config

    @classmethod
    def get_mdi_engine_command(
        cls,
        executor,
        seamm_options,
        *,
        method,
        port,
        hostname="localhost",
        charge=0,
        multiplicity=1,
        n_atoms=None,
        engine_name="XNN",
        extra_args=None,
        **kwargs,
    ):
        """Build the argv that launches the xnn MDI *engine* over TCP.

        The driver (e.g. the Energy step, via ``seamm_mdi.MDIEngine``) owns the
        rendezvous (TCP, ``port``, ``hostname``) and passes it in; everything
        xnn-specific -- the conda environment, the checkpoint path for the model,
        the torch device -- is supplied here, so the driver hardwires no xnn
        knowledge. The atom count, atomic numbers, coordinates and cell all
        arrive over the MDI handshake from the driver.

        Parameters
        ----------
        executor, seamm_options
            Passed straight to ``get_executor_config``.
        method : str
            The model name, as advertised by ``get_model_chemistry_options``.
        port : int
            TCP port the engine dials; chosen by the driver.
        hostname : str
            Host the engine dials.
        charge, multiplicity : int
            Accepted for interface compatibility; an MLFF has no notion of them.
        n_atoms : int, optional
            Number of atoms (accepted for interface compatibility).
        engine_name : str
            The MDI ``-name`` for the engine (default "XNN").
        extra_args : list of str, optional
            Extra ``xnn mdi`` flags appended verbatim (e.g. ``--dtype float32``).

        Returns
        -------
        list of str
            A ready-to-run argv, possibly with a leading ``OMP_NUM_THREADS=n``
            environment prefix (honored by ``seamm_mdi.MDIEngine`` and by
            ``shlex.join`` into a launch script).
        """
        config = cls.get_executor_config(executor, seamm_options)

        models = cls.available_models(config)
        if method not in models:
            raise ValueError(
                f"'{method}' is not an available xnn model; the models found are "
                f"{sorted(models)}. Check the 'models' directories in xnn.ini."
            )
        checkpoint = models[method]

        installation = config.get("installation", "conda")
        code = config.get("code", "xnn").split()
        if installation == "conda":
            environment = config["conda-environment"]
            if environment[0] == "~" or Path(environment).is_absolute():
                env_args = ["-p", str(Path(environment).expanduser())]
            else:
                env_args = ["-n", environment]
            argv = [config["conda"], "run", "--live-stream", *env_args, *code]
        elif installation == "local":
            argv = [*code]
        else:
            raise NotImplementedError(
                "The xnn MDI engine is wired up for 'conda' and 'local' "
                f"installations; xnn.ini selects '{installation}'."
            )

        mdi_init = (
            f"-role ENGINE -name {engine_name} -method TCP "
            f"-port {port} -hostname {hostname}"
        )
        argv += [
            "mdi",
            "--ckpt",
            str(checkpoint),
            "--device",
            config.get("device", "cpu").strip() or "cpu",
            "-mdi",
            mdi_init,
        ]
        if extra_args:
            argv.extend(extra_args)

        # Cap the CPU threads from the [xnn-step] section of seamm.ini, if asked.
        n = cls.thread_count()
        if n is not None:
            argv = [f"OMP_NUM_THREADS={n}", f"MKL_NUM_THREADS={n}", *argv]

        return argv

    @staticmethod
    def seamm_ini_path():
        """Path to the global ``seamm.ini``: ``~/.seamm.d/seamm.ini``, or
        ``$SEAMM_INI`` if set (handy for testing against an alternate config)."""
        override = os.environ.get("SEAMM_INI")
        if override:
            return Path(override).expanduser()
        return Path.home() / ".seamm.d" / "seamm.ini"

    @classmethod
    def thread_count(cls):
        """The CPU thread cap for the engine from ``[xnn-step] ncores`` in
        seamm.ini, or ``None`` for 'available' (PyTorch's default)."""
        ncores = None
        try:
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read(cls.seamm_ini_path())
            if cfg.has_section("xnn-step"):
                ncores = cfg["xnn-step"].get("ncores", None)
        except Exception as e:  # never let config parsing break a run
            logger.warning(f"Could not read [xnn-step] from seamm.ini: {e}")
        if ncores in (None, "", "available"):
            return None
        try:
            return max(1, int(ncores))
        except (TypeError, ValueError):
            return None

    def __init__(self, flowchart=None, gui=None):
        """Initialize this helper class, which is used by
        the application via stevedore to get information about
        and create node objects for the flowchart
        """
        pass

    def description(self):
        """Return a description of what this extension does."""
        return XnnStep.my_description

    def create_node(self, flowchart=None, **kwargs):
        """Create and return the new node object.

        Parameters
        ----------
        flowchart: seamm.Node
            A non-graphical SEAMM node

        **kwargs : keyworded arguments
            Various keyworded arguments such as title, namespace or
            extension representing the title displayed in the flowchart,
            the namespace for the plugins of a subflowchart and
            the extension, respectively.

        Returns
        -------
        Xnn
        """

        return xnn_step.Xnn(flowchart=flowchart, **kwargs)

    def create_tk_node(self, canvas=None, **kwargs):
        """Create and return the graphical Tk node object.

        Parameters
        ----------
        canvas : tk.Canvas
            The Tk Canvas widget

        **kwargs : keyworded arguments
            Various keyworded arguments such as tk_flowchart, node, canvas,
            x, y, w, h, etc.

        Returns
        -------
        TkXnn
        """

        return xnn_step.TkXnn(canvas=canvas, **kwargs)

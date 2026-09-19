Distributing MLFFs: sidecar and catalog (draft 2026-09-18)
==========================================================

Status: **draft for discussion** (Paul, Sina). Nothing here is implemented yet.

Why
---
Trained checkpoints are megabytes to gigabytes, so they do not belong in a
plug-in's ``data/`` directory on GitHub. Publishing each model on Zenodo gives
it a DOI (a *concept* DOI that always resolves to the newest version, plus one
DOI per version), long-term storage, and a citation. SEAMM then needs three
things: a way to *describe* a model (the sidecar), a way to *find* the models we
recommend (the catalog), and rules for *caching, updating and citing* them.

Requirements gathered so far
----------------------------
* The curated catalog must be **easy to update by hand, several times a day if
  needed** -- the packaging workflow depends on it. This rules out putting the
  catalog itself in a Zenodo record (every edit would be a new immutable
  version). The big files go on Zenodo; the small catalog lives in a Git repo
  and is fetched as a raw file.
* Offline and secure sites: a model fetched on one machine, or from the Zenodo
  web page by hand, must be fully usable (and citable) when copied to a machine
  without Internet access. Hence the sidecar travels with the checkpoint.
* The model-chemistry *name* stays the plain model name (``xnn:MLFF@water``);
  which version and which source were used are reported and cited, not encoded
  in the name, so a retrain does not change anyone's flowchart.
* xnn checkpoints already carry some of this internally (``{"model":
  state_dict, "cfg": Config}``: model family, cutoff, architecture, the species
  list in ``cfg.model.extra``, the training paths and property keys). Whatever
  can be derived from the checkpoint should be *derived*, not retyped; the
  sidecar adds what the checkpoint cannot know (DOI, provenance, conventions,
  licence). Open question for Sina: is ``cfg`` stable enough to read from
  outside the trainer, and can xnn write the sidecar itself at ``save`` time?

The sidecar
-----------
One JSON file next to each checkpoint, same stem, extension ``.mlff.json``
(``water_mace_les.pt`` + ``water_mace_les.mlff.json``). It is what SEAMM reads;
the checkpoint is opaque to SEAMM. Fields marked *derivable* can be filled from
the checkpoint by the publishing utility; the rest are supplied by the author.

.. code-block:: json

   {
     "schema": "seamm-mlff-sidecar/1",

     "name": "water_mace_les",
     "title": "Water MACE-LES, run 56 + trimers",
     "description": "MACE with long-range electrostatics (LES) trained on ...",
     "version": "2026.09.15",
     "created": "2026-09-15T19:37:00Z",
     "authors": [
       {"name": "Sina Mostafanejad", "affiliation": "MolSSI", "orcid": "0000-..."}
     ],
     "license": "CC-BY-4.0",

     "doi": "10.5281/zenodo.1234567",
     "concept_doi": "10.5281/zenodo.1234566",
     "zenodo_record": 1234567,
     "files": {
       "water_mace_les.pt": {"sha256": "…", "size": 3653832}
     },

     "engine": {
       "code": "xnn",
       "min_version": "0.1.0",
       "family": "mace",
       "dtype": "float32",
       "cutoff": 6.0,
       "cutoff_units": "Å",
       "checkpoint_format": "xnn-trainer/1"
     },

     "applicability": {
       "elements": ["H", "O"],
       "periodic": true,
       "charges": [0],
       "spin_multiplicities": [1],
       "phases": ["gas-phase clusters", "liquid water"],
       "notes": "Trained on 1- to 3-mers plus liquid snapshots; not for ions."
     },

     "conventions": {
       "energy_units": "eV",
       "length_units": "Å",
       "energy_reference": "formation energy, isolated-atom reference, E0 = 0",
       "reference_model_chemistry": "ORCA:DFT@revDSD-PBEP86-D4/def2-TZVPPD",
       "counterpoise": "SSFC",
       "stress_sign": "pressure (MDI convention)"
     },

     "training": {
       "data_doi": "10.5281/zenodo.7654321",
       "n_structures": 8666,
       "energy_mae": {"value": 0.4, "units": "meV/atom"},
       "force_mae": {"value": 12.0, "units": "meV/Å"},
       "test_set": "held-out 10 %"
     },

     "citations": [
       {"doi": "10.5281/zenodo.1234567", "note": "This model."},
       {"doi": "10.48550/arXiv.2206.07697", "note": "MACE architecture."}
     ]
   }

Field notes
~~~~~~~~~~~
* ``schema`` lets the reader reject or upgrade old sidecars.
* ``name`` is the model-chemistry method token; it must avoid ``: @ / |`` and
  should be stable across versions. ``version`` is free text but a date is
  recommended, matching SEAMM's CalVer habit.
* ``doi`` / ``concept_doi`` / ``zenodo_record``: the version DOI is what a job
  cites; the concept DOI is what the update check follows. All three are absent
  for a private, unpublished model -- the sidecar is still useful.
* ``files`` with ``sha256``: integrity after a manual copy, and detection of a
  model that changed without a version bump.
* ``engine``: everything the launcher and the driver need to know without
  loading torch -- the family, dtype (mps needs float32), cutoff, and the
  minimum xnn that can unpickle it. *Derivable* from ``cfg`` except
  ``min_version``.
* ``applicability``: what the Model Chemistry step can filter on (elements,
  periodic) and what the Energy step should warn about at run time (a charged or
  open-shell configuration for a model trained on neutral singlets). ``elements``
  is derivable from the species list in ``cfg.model.extra``.
* ``conventions``: the campaign decisions a consumer must know to compare with
  QM -- units, and above all the **energy reference** (per-atom formation energy
  with E0 = 0, per the water/electrolyte plan). Without this a REF_energy from
  the model and one from ORCA cannot be put on the same axis.
* ``training``: provenance and headline accuracy; the ``data_doi`` closes the
  loop to the training set on Zenodo.
* ``citations``: what ``self.references.cite`` should add when the model is
  used, beyond the plug-in's own reference.

The catalog
-----------
A single JSON file in a small Git repository (proposed
``molssi-seamm/mlff-catalog``, file ``catalog.json`` on ``main``), fetched by
raw URL. Editing it is a commit -- by hand or by the publishing utility -- so
it can change as often as needed, with history and review if wanted. Zenodo
holds the files; the catalog holds pointers plus a copy of each model's sidecar
so the Model Chemistry dialog can list and filter models without touching
Zenodo at all.

.. code-block:: json

   {
     "schema": "seamm-mlff-catalog/1",
     "updated": "2026-09-18T21:00:00Z",
     "models": {
       "water_mace_les": {
         "concept_doi": "10.5281/zenodo.1234566",
         "latest": "2026.09.15",
         "versions": {
           "2026.09.15": {
             "doi": "10.5281/zenodo.1234567",
             "zenodo_record": 1234567,
             "files": {"water_mace_les.pt": {"sha256": "…", "size": 3653832}},
             "sidecar": { "...the full sidecar above..." }
           },
           "2026.08.08": {
             "doi": "10.5281/zenodo.1234500",
             "zenodo_record": 1234500,
             "files": {"water_mace_les.pt": {"sha256": "…", "size": 3653832}},
             "sidecar": { "..." },
             "deprecated": "Superseded by 2026.09.15 (trimer data added)."
           }
         }
       }
     }
   }

* Keyed by model name; ``latest`` names the version the plug-in offers by
  default; older versions remain resolvable for reproducibility, optionally
  marked ``deprecated`` with a reason.
* The embedded ``sidecar`` is a copy: the authoritative one is the file on
  Zenodo. The utility that adds a version to the catalog copies it in, so the
  two cannot drift by hand-editing unless someone edits the copy on purpose.
* The plug-in caches the catalog at ``~/.seamm.d/xnn/catalog.json`` with the
  HTTP ETag, refreshing at most once an hour when online and silently keeping
  the cached copy when not. A user can also point ``catalog`` in ``xnn.ini`` at
  a file or a different URL (a site mirror behind a firewall).

Resolution and caching
----------------------
Model directories keep today's order (``personal:xnn``, ``local:xnn``, extra
directories), and the catalog is appended as a further, remote source. When a
model chemistry ``xnn:MLFF@<name>`` is used:

1. Look for ``<name>.pt`` in the model directories. If found, read its
   sidecar (if present) and use it. A checkpoint without a sidecar is still
   usable, reported as *unpublished/local*, and not citable.
2. Otherwise look the name up in the catalog and **download** the ``latest``
   version's files and sidecar into ``local:xnn`` (machine-wide) -- or
   ``personal:xnn`` if ``local`` is not writable -- verifying the sha256.
   Downloading happens at run time, once; the job then proceeds as with a local
   file. Model Chemistry shows catalog-only models with a marker such as
   "(download)".
3. Report and cite: the output names the file used and its version DOI, and
   ``self.references.cite`` adds the model's citations.

Update policy: default **use what is cached**. When online and the catalog says
a newer version exists, print a one-line notice; never swap under a running
campaign. Explicit actions: ``fetch --update`` to take the newest, or pin a
version in ``xnn.ini`` (``water_mace_les = 2026.08.08``) for a study that must
not move.

Utility (offline and secure sites)
----------------------------------
Subcommands of ``xnn-step-installer`` (or a ``seamm-mlff`` script if this
becomes shared with other MLFF providers):

* ``models list [--catalog URL|FILE]`` -- what the catalog offers, with
  version, elements and whether it is cached.
* ``models fetch <name> [--version V] [--to DIR]`` -- download checkpoint +
  sidecar, verify sha256. Run on a connected machine, copy the two files to the
  inside machine's ``local:xnn`` or ``personal:xnn``; nothing else is needed.
* ``models check`` -- compare cached versions with the catalog.
* ``models publish <checkpoint> --sidecar X.json [--sandbox]`` -- upload to
  Zenodo (via ``seamm_util.Zenodo``, which already handles deposits, versions
  and publishing), then add the version to the catalog file for commit. Fills
  the derivable sidecar fields from the checkpoint.

Open questions
--------------
* Sina: what exactly does ``cfg`` expose (species list location, units
  assumptions), and would xnn write the sidecar at ``Trainer.save``?
* Zenodo community for the models (``seamm`` exists for flowcharts?) and the
  licence to standardise on.
* Whether the catalog repository should also carry the training-set records,
  or those stay per-model via ``training.data_doi`` only.
* Generalise now or later: the sidecar and catalog are engine-agnostic (only
  ``engine.code`` differs); a MACE-native or other provider could share them,
  which argues for the code living in a small shared package rather than in
  ``xnn_step``.

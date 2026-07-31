"""terratorch custom-module registration for gelos-lc project backbones.

terratorch auto-imports this package from the working directory at startup
(``terratorch/registry/custom_registry.py``). Backbone registrations that
belong to the shared ``gelos`` library live in ``gelos/backbones/`` and are
imported from ``gelos/generation.py`` instead — this file is reserved for
any per-project backbones specific to gelos-lc.
"""

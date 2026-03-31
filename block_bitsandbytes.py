"""
Import hook that blocks bitsandbytes on RDNA2 to prevent silent segfaults.

Usage — add to the TOP of your launch script or sitecustomize.py:
    import block_bitsandbytes  # noqa: F401

Or let launch.sh inject it via PYTHONPATH.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys


class _BitsAndBytesBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Intercepts `import bitsandbytes` and raises a clear error."""

    BLOCKED = {"bitsandbytes"}

    def find_module(self, fullname, path=None):
        # Block the top-level package and any submodule.
        top = fullname.split(".")[0]
        if top in self.BLOCKED:
            return self
        return None

    def load_module(self, fullname):
        raise ImportError(
            f"'{fullname}' is blocked on RDNA2 (gfx1030). "
            "bitsandbytes / adamw8bit are CUDA-only and will segfault on AMD HIP. "
            "Use adafactor instead.  To unblock: unset BITSANDBYTES_BLOCKED"
        )


import os as _os

if _os.environ.get("BITSANDBYTES_BLOCKED", "0") == "1":
    sys.meta_path.insert(0, _BitsAndBytesBlocker())

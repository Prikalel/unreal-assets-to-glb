#!/usr/bin/env python3
"""Developer launcher.

This thin wrapper exists so the existing ``python main.py ./Input`` workflow
keeps working during development. When the package is installed, use the
console command instead::

    unreal-assets-to-glb ./Input

This file is intentionally NOT shipped in the built wheel/sdist (it is not a
declared py-module); the real entry point lives in :mod:`uasset.cli`.
"""
from uasset.cli import main

if __name__ == "__main__":
    main()

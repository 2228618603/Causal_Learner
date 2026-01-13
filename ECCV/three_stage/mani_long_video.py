#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility wrapper for the three-stage long-video pipeline.

This lives under `ECCV/three_stage/` so the original two-stage scripts in `ECCV/`
remain untouched. It forwards to `three_stage/pipeline.py`.
"""

from pipeline import main


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Usage: ./trustyai-shields-fms-distribution/build.py

import sys
from pathlib import Path

# Add parent directory to Python path to import shared_build
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_build import build_distribution


def main():
    build_distribution("trustyai-shields-fms-distribution", "TrustyAI Shields FMS")


if __name__ == "__main__":
    main()

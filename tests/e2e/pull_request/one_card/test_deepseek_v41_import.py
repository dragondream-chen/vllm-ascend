# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise model inspection imports without the unit-test import bootstrap."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "vllm_ascend.device.device_op",
        "vllm_ascend.models.deepseek_v41.model",
        "vllm_ascend.models.deepseek_v41.dspark",
    ],
)
def test_deepseek_v41_fresh_process_import(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

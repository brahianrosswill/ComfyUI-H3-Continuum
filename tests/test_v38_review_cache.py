from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
)
from ComfyUI_H3_Continuum_Join.v3.easy_nodes import H3ContinuumEasyV38
from ComfyUI_H3_Continuum_Join.v3.review_control import (
    GENERATION_MODE_FULL_RUN,
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
    REVIEW_ACTION_FINISH_REMAINING,
    REVIEW_ACTION_REGENERATE_CURRENT,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = Path(
    os.environ.get(
        "H3_CONTINUUM_COMFY_ROOT",
        r"D:\StabilityMatrix\Data\Packages\ComfyUI_W",
    )
)


@pytest.mark.parametrize(
    "review_action",
    (
        REVIEW_ACTION_CONTINUE,
        REVIEW_ACTION_REGENERATE_CURRENT,
        REVIEW_ACTION_FINISH_REMAINING,
    ),
)
def test_v38_review_is_always_changed_for_every_review_action(review_action):
    first = H3ContinuumSamplerV38.IS_CHANGED(
        generation_mode=GENERATION_MODE_REVIEW,
        review_action=review_action,
        unrelated_current_input="accepted through Core kwargs",
    )
    second = H3ContinuumSamplerV38.IS_CHANGED(
        generation_mode=GENERATION_MODE_REVIEW,
        review_action=review_action,
    )
    assert math.isnan(first)
    assert math.isnan(second)
    assert first != second


@pytest.mark.parametrize(
    "review_action",
    (
        REVIEW_ACTION_CONTINUE,
        REVIEW_ACTION_REGENERATE_CURRENT,
        REVIEW_ACTION_FINISH_REMAINING,
    ),
)
def test_v38_full_run_keeps_stable_core_cache_behavior(review_action):
    assert H3ContinuumSamplerV38.IS_CHANGED(
        generation_mode=GENERATION_MODE_FULL_RUN,
        review_action=review_action,
    ) is False


def test_v37_and_easy_do_not_inherit_the_v38_cache_contract():
    assert "IS_CHANGED" not in H3ContinuumSamplerV37.__dict__
    assert "IS_CHANGED" not in H3ContinuumEasyV38.__dict__
    assert not hasattr(H3ContinuumSamplerV37, "IS_CHANGED")
    assert not hasattr(H3ContinuumEasyV38, "IS_CHANGED")


def test_real_comfy_cache_signature_is_stable_only_for_full_run(tmp_path):
    comfy_root = DEFAULT_COMFY_ROOT
    if not (comfy_root / "execution.py").is_file():
        pytest.skip("A real ComfyUI runtime is required for the Core cache harness")

    harness = tmp_path / "v38-review-core-cache-harness.py"
    harness.write_text(
        textwrap.dedent(
            f"""
            import asyncio
            import importlib.util
            import json
            import os
            from pathlib import Path
            import sys

            def block_cuda(*args, **kwargs):
                raise BaseException("CPU cache harness requested CUDA initialization")

            def early_profile(frame, event, arg):
                if event == "c_call" and getattr(arg, "__name__", "") == "_cuda_init":
                    block_cuda()

            sys.setprofile(early_profile)
            import torch
            assert not torch.cuda.is_initialized()
            torch.cuda._lazy_init = block_cuda
            torch._C._cuda_init = block_cuda
            sys.setprofile(None)

            comfy_root = Path({str(comfy_root)!r})
            plugin_root = Path({str(ROOT)!r})
            os.chdir(comfy_root)
            sys.path.insert(0, str(comfy_root))

            # Core imports must stay CPU-only; Kitchen's import-time INT8
            # capability check otherwise asks for GPU capability even in CPU mode.
            sys.argv = [str(__file__), "--cpu"]
            import comfy.options
            comfy.options.enable_args_parsing()
            from comfy.cli_args import args
            assert args.cpu
            # This harness tests Core cache keys, not a GPU attention kernel.
            # Optional SageAttention probes GPU capability at import time even
            # with Core --cpu. Model an absent optional extension in this child
            # only; leave installed packages and all CUDA tripwires unchanged.
            sys.modules["sageattention"] = None
            if importlib.util.find_spec("comfy_kitchen") is not None:
                import comfy_kitchen
                comfy_kitchen.int8_attention_is_available = lambda *a, **k: False

            import nodes as comfy_nodes
            from comfy_execution.caching import CacheKeySetInputSignature
            from comfy_execution.graph import DynamicPrompt
            from execution import IsChangedCache

            package_name = "H3ContinuumPhaseFR1Source"
            spec = importlib.util.spec_from_file_location(
                package_name,
                plugin_root / "__init__.py",
                submodule_search_locations=[str(plugin_root)],
            )
            package = importlib.util.module_from_spec(spec)
            package.__path__ = [str(plugin_root)]
            sys.modules[package_name] = package
            spec.loader.exec_module(package)

            class_type = "H3ContinuumSamplerV38PhaseFR1Probe"
            comfy_nodes.NODE_CLASS_MAPPINGS[class_type] = (
                package.NODE_CLASS_MAPPINGS["H3ContinuumSamplerV38"]
            )

            async def signature(mode, action):
                prompt = {{
                    "305": {{
                        "class_type": class_type,
                        "inputs": {{
                            "generation_mode": mode,
                            "review_action": action,
                        }},
                    }}
                }}
                dynprompt = DynamicPrompt(prompt)
                changed = IsChangedCache("phase-f-r1", dynprompt, None)
                keys = CacheKeySetInputSignature(dynprompt, ["305"], changed)
                await keys.add_keys(["305"])
                return keys.get_data_key("305")

            async def main():
                full_1 = await signature("Full Run", "Continue / Next")
                full_2 = await signature("Full Run", "Continue / Next")
                review_continue_1 = await signature(
                    "Review Each Chunk", "Continue / Next"
                )
                review_continue_2 = await signature(
                    "Review Each Chunk", "Continue / Next"
                )
                review_regen_1 = await signature(
                    "Review Each Chunk", "Regenerate Current"
                )
                review_regen_2 = await signature(
                    "Review Each Chunk", "Regenerate Current"
                )
                review_finish_1 = await signature(
                    "Review Each Chunk", "Finish Remaining"
                )
                review_finish_2 = await signature(
                    "Review Each Chunk", "Finish Remaining"
                )
                print(json.dumps({{
                    "full_stable": full_1 == full_2,
                    "review_continue_changed": review_continue_1 != review_continue_2,
                    "review_regenerate_changed": review_regen_1 != review_regen_2,
                    "review_finish_changed": review_finish_1 != review_finish_2,
                    "cpu_mode": args.cpu,
                    "cuda_initialized": torch.cuda.is_initialized(),
                }}))

            asyncio.run(main())
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(harness)],
        cwd=comfy_root,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "full_stable": True,
        "review_continue_changed": True,
        "review_regenerate_changed": True,
        "review_finish_changed": True,
        "cpu_mode": True,
        "cuda_initialized": False,
    }

"""Lightweight installation verifier for a local ComfyUI environment."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


EXPECTED_PUBLIC_NODE_DISPLAY_NAMES = {
    "H3ContinuumSamplerV38": "H3 Continuum Sampler V3.8",
    "H3ContinuumReferenceAudios": "H3 Continuum Reference Audios",
    "H3ContinuumAssembleSeamV35": "H3 Continuum Finalize",
    "H3EasyLoadImage": "H3 Continuum Load Image",
    "H3EasyLoadAudio": "H3 Continuum Load Audio",
    "H3ContinuumLoadVideo": "H3 Continuum Load Video",
    "H3ContinuumSecondPassV35": "H3 Continuum Second Pass",
}


def load_package(root: Path):
    name = "h3_continuum_join_verify"
    spec = importlib.util.spec_from_file_location(
        name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create package import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def public_registration_issues(module) -> list[str]:
    """Return exact public-surface differences for the installed V3.8 package."""

    expected = EXPECTED_PUBLIC_NODE_DISPLAY_NAMES
    actual_classes = set(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    expected_classes = set(expected)
    issues: list[str] = []
    if actual_classes != expected_classes:
        missing = sorted(expected_classes - actual_classes)
        extra = sorted(actual_classes - expected_classes)
        if missing:
            issues.append(f"missing public node IDs: {missing}")
        if extra:
            issues.append(f"unexpected public node IDs: {extra}")

    actual_names = dict(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    for node_id, display_name in expected.items():
        actual_name = actual_names.get(node_id)
        if actual_name != display_name:
            issues.append(
                f"display name mismatch for {node_id}: "
                f"expected {display_name!r}, got {actual_name!r}"
            )
    unexpected_names = sorted(set(actual_names) - expected_classes)
    if unexpected_names:
        issues.append(f"unexpected display-name IDs: {unexpected_names}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    comfy_root = args.comfy_root.resolve()
    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(comfy_root))

    module = load_package(package_root)
    from h3_continuum_join_verify.compatibility import (
        check_comfy_h3_runtime,
        run_native_layout_self_test,
    )
    from h3_continuum_join_verify.constants import PROMPT_MODE_FIXED
    from h3_continuum_join_verify.v2.prompts import make_prompt_plan
    from h3_continuum_join_verify.version import PACKAGE_VERSION

    issues = check_comfy_h3_runtime()
    if issues:
        print("H3 Continuum Join runtime verification FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    try:
        layout_report = run_native_layout_self_test()
    except Exception as exc:
        print(f"Native PackedLayout self-test FAILED: {exc}")
        return 1

    try:
        prompt_plan = make_prompt_plan(
            mode=PROMPT_MODE_FIXED,
            script="Continuum runtime verification prompt",
            chunks=3,
            chunk_seconds=5.0,
        )
        if len(prompt_plan["prompts"]) != 3 or len(set(prompt_plan["hashes"])) != 1:
            raise RuntimeError("unexpected Fixed prompt-plan result")
    except Exception as exc:
        print(f"V2 prompt/session core self-test FAILED: {exc}")
        return 1

    registration_issues = public_registration_issues(module)
    if registration_issues:
        print("H3 Continuum V3.8 public-node verification FAILED:")
        for issue in registration_issues:
            print(f"  - {issue}")
        return 1
    actual = set(module.NODE_CLASS_MAPPINGS)
    print(f"H3 Continuum {PACKAGE_VERSION} runtime verification passed.")
    print(layout_report)
    print(f"V{PACKAGE_VERSION} Fixed 3x5s prompt-plan self-test passed.")
    print("Registered nodes:")
    for key in sorted(actual):
        print(f"  - {module.NODE_DISPLAY_NAME_MAPPINGS[key]}")
    print("GPU checkpoint generation was not run by this lightweight verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

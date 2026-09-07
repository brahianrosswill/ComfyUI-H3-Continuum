"""Run Storage identity must not track Core's normal loading lifecycle."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from ComfyUI_H3_Continuum_Join.run_storage import (
    _hash, _module_signature, _observe, _video_vae_signature,
)


def vae_fixture():
    module = torch.nn.Module()
    module.encoder = torch.nn.Linear(3, 4, bias=False)
    module.decoder = torch.nn.Module()
    module.decoder.register_parameter('scale1', torch.nn.Parameter(torch.tensor([0.123456, 0.654321, 1.234567])))
    module.decoder.proj = torch.nn.Linear(4, 3, bias=False)
    module.decoder.scale1_comfy_model_dtype = torch.float16
    return SimpleNamespace(first_stage_model=module, patcher=None, vae_dtype=torch.float16)


def test_video_vae_identity_survives_core_lazy_cast_without_mutation():
    vae = vae_fixture()
    before_tensors = {n: p.detach().clone() for n, p in vae.first_stage_model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    cold, cold_safe = _video_vae_signature(vae, required=True)
    assert cold_safe
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(before_tensors[n], p) for n, p in vae.first_stage_model.named_parameters())
    assert vae.first_stage_model.decoder.scale1.dtype == torch.float32
    # Same assignment as Core DynamicVRAM load, not a Continuum mutation.
    vae.first_stage_model.decoder.scale1 = torch.nn.Parameter(vae.first_stage_model.decoder.scale1.detach().half())
    warm, warm_safe = _video_vae_signature(vae, required=True)
    assert warm_safe and cold == warm
    assert not torch.cuda.is_initialized()


def test_video_vae_real_weight_change_is_still_detected():
    vae = vae_fixture()
    before, _ = _video_vae_signature(vae, required=True)
    with torch.no_grad():
        vae.first_stage_model.decoder.scale1.add_(0.25)
    after, _ = _video_vae_signature(vae, required=True)
    assert before != after


def test_declared_precision_change_is_not_ignored():
    vae = vae_fixture()
    before, _ = _video_vae_signature(vae, required=True)
    vae.first_stage_model.decoder.scale1_comfy_model_dtype = torch.bfloat16
    after, _ = _video_vae_signature(vae, required=True)
    assert before != after


def test_no_core_dtype_metadata_preserves_original_probe_contract():
    module = torch.nn.Linear(3, 2, bias=False)
    before, safe = _module_signature(module)
    assert safe and before['weight_probe'][0]['dtype'] == 'torch.float32'
    after, safe = _module_signature(copy.deepcopy(module).half())
    assert safe and before != after


def test_set_identity_is_complete_order_independent_and_typed():
    keys = {f'layer.{i}.lora_A.weight' for i in range(500)}
    before = copy.deepcopy(keys)
    a, safe = _observe(keys)
    b, other_safe = _observe(set(reversed(sorted(keys))))
    assert safe and other_safe and a == b and keys == before
    assert len(a['items']) == 500
    changed, _ = _observe(keys | {'different.weight'})
    frozen, _ = _observe(frozenset(keys))
    assert a != changed and a != frozen
    assert _observe([1, 2])[0] != _observe([2, 1])[0]


def test_adapter_weight_and_strength_changes_are_not_hidden_by_set_fix():
    adapter = SimpleNamespace(loaded_keys={'b', 'a'}, weights=[torch.arange(8.), 8.])
    before, safe = _observe([(1., adapter)])
    assert safe
    adapter.weights[0][0] += 1
    after, safe = _observe([(1., adapter)])
    assert safe and before != after
    assert after != _observe([(0.5, adapter)])[0]


def test_lora_set_identity_matches_across_python_hash_seeds():
    root = str(Path(__file__).resolve().parents[1])
    code = f'''
import sys, types, json
p = types.ModuleType("ComfyUI_H3_Continuum_Join")
p.__path__ = [{root!r}]
sys.modules[p.__name__] = p
from ComfyUI_H3_Continuum_Join.run_storage import _observe, _hash
from types import SimpleNamespace
import torch
obj = SimpleNamespace(loaded_keys={{f"layer.{{i}}.lora_A.weight" for i in range(500)}}, weights=[torch.arange(8.), 8.])
value, safe = _observe(obj)
assert safe and not torch.cuda.is_initialized()
print("RESULT:" + json.dumps({{"sha": _hash(value), "safe": safe}}))
'''
    results = []
    for seed in ('1', '777'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        run = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=30, check=True)
        records = [line[7:] for line in run.stdout.splitlines() if line.startswith('RESULT:')]
        assert len(records) == 1, run.stdout
        results.append(json.loads(records[0]))
    assert results[0] == results[1]

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import inspect
from pathlib import Path
import threading
import weakref

import pytest
import torch

from ComfyUI_H3_Continuum_Join import reference, reference_audio, reference_video
from ComfyUI_H3_Continuum_Join.reference import ReferenceAssets
from ComfyUI_H3_Continuum_Join.reference_audio import (
    ReferenceAudioAssets,
    ReferenceAudioSource,
)
from ComfyUI_H3_Continuum_Join.temporal import video_latent_t
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes
from ComfyUI_H3_Continuum_Join.v3.ref_encode_cache import (
    RefEncodeCache,
    clear_ref_encode_cache,
    format_ref_encode_cache_diagnostics,
    get_ref_encode_cache,
    inspect_ref_encode_cache,
    make_ref_encode_cache_key,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_global_cache():
    clear_ref_encode_cache()
    yield
    clear_ref_encode_cache()


class ImageVAE:
    def __init__(self):
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        value = float(image.mean().item())
        return torch.full((1, 24, 7, 2, 2), value, dtype=torch.float32)


class VideoVAE:
    def __init__(self):
        self.calls = 0

    def encode(self, frames):
        self.calls += 1
        latent_t = video_latent_t(int(frames.shape[0]))
        value = float(frames.mean().item())
        return torch.full((1, 24, latent_t, 2, 2), value, dtype=torch.float32)


class AudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.calls = 0

    def encode(self, waveform):
        self.calls += 1
        return torch.full(
            (1, 32, 2, 9),
            float(waveform.mean().item()),
            dtype=torch.float32,
        )


def _key(identity: str, *, kind: str = "reference_image"):
    return make_ref_encode_cache_key(kind, 1, identity)


def _image_assets(*hashes: str) -> ReferenceAssets:
    images = tuple(
        torch.full((1, 32, 32, 3), float(index), dtype=torch.float32)
        for index in range(len(hashes))
    )
    return ReferenceAssets(
        images=images,
        latents=tuple(None for _ in images),
        image_hashes=tuple(hashes),
        combined_hash="combined",
        size_mode=reference.REFERENCE_SIZE_MATCH_OUTPUT,
    )


def _video_source(
    *,
    frames: int = 40,
    target_frames: int = 22,
    size_mode: str = reference_video.REFERENCE_VIDEO_SIZE_EFFICIENT,
    value: float = 0.25,
):
    return reference_video.prepare_reference_video_source(
        torch.full((frames, 32, 32, 3), value, dtype=torch.float32),
        target_frames=target_frames,
        output_width=64,
        output_height=64,
        size_mode=size_mode,
    )


def _audio_source(identity: str = "audio-a") -> ReferenceAudioSource:
    waveform = torch.zeros((1, 2, 3200), dtype=torch.float32)
    return ReferenceAudioSource(
        waveform=waveform,
        source_sample_rate=32000,
        source_shape=tuple(waveform.shape),
        source_dtype=str(waveform.dtype),
        source_sha256=f"source-{identity}",
        resolved_vae_sample_rate=32000,
        resampled_shape=tuple(waveform.shape),
        resampled_dtype=str(waveform.dtype),
        resampled_sha256=f"resampled-{identity}",
        combined_hash=identity,
    )


def test_cache_core_miss_store_hit_returns_private_clone():
    cache = RefEncodeCache()
    vae = ImageVAE()
    key = _key("a")
    source = torch.arange(8, dtype=torch.float32)
    events = []

    assert cache.lookup(vae, key, event_sink=events.append) is None
    assert cache.store(vae, key, source, event_sink=events.append)
    hit = cache.lookup(vae, key, event_sink=events.append)

    assert [event.action for event in events] == ["miss", "store", "hit"]
    assert torch.equal(hit, source)
    assert hit is not source
    assert hit.data_ptr() != source.data_ptr()
    hit.add_(100)
    assert torch.equal(cache.lookup(vae, key), source)


def test_cache_namespace_requires_same_vae_object():
    cache = RefEncodeCache()
    first = ImageVAE()
    second = ImageVAE()
    key = _key("same")
    cache.store(first, key, torch.ones(2))

    assert cache.lookup(first, key) is not None
    assert cache.lookup(second, key) is None


def test_cache_lru_evicts_global_oldest_entry():
    cache = RefEncodeCache(max_entries=2, max_bytes=1024)
    vae = ImageVAE()
    first, second, third = _key("1"), _key("2"), _key("3")
    cache.store(vae, first, torch.ones(1))
    cache.store(vae, second, torch.ones(1) * 2)
    assert cache.lookup(vae, first) is not None
    cache.store(vae, third, torch.ones(1) * 3)

    assert cache.lookup(vae, first) is not None
    assert cache.lookup(vae, second) is None
    assert cache.lookup(vae, third) is not None
    assert cache.inspect()["resident_entries"] == 2


def test_cache_byte_limit_and_oversize_bypass():
    cache = RefEncodeCache(max_entries=16, max_bytes=16)
    vae = ImageVAE()
    events = []

    assert not cache.store(
        vae,
        _key("too-large"),
        torch.ones(5, dtype=torch.float32),
        event_sink=events.append,
    )
    assert events[-1].action == "bypass"
    assert events[-1].reason == "entry_exceeds_byte_limit"
    assert cache.inspect()["resident_entries"] == 0


def test_cache_clear_resets_entries_and_stats():
    cache = RefEncodeCache()
    vae = ImageVAE()
    cache.store(vae, _key("a"), torch.ones(1))
    cache.lookup(vae, _key("a"))
    cache.clear()

    assert cache.inspect()["resident_entries"] == 0
    assert cache.inspect()["stats"] == {}


def test_weak_vae_cleanup_releases_namespace():
    cache = RefEncodeCache()
    vae = ImageVAE()
    cache.store(vae, _key("a"), torch.ones(1))
    vae_ref = weakref.ref(vae)
    del vae
    gc.collect()

    assert vae_ref() is None
    assert cache.inspect()["resident_entries"] == 0
    assert cache.inspect()["vae_namespaces"] == 0


@pytest.mark.parametrize("vae", [object(), []])
def test_unweakrefable_or_unhashable_vae_is_bypassed(vae):
    cache = RefEncodeCache()
    events = []

    assert cache.lookup(vae, _key("a"), event_sink=events.append) is None
    assert events[-1].action == "bypass"
    assert events[-1].reason == "vae_not_weakrefable_or_hashable"


def test_concurrent_hits_do_not_share_returned_tensor_or_corrupt_cache():
    cache = RefEncodeCache()
    vae = ImageVAE()
    key = _key("threaded")
    expected = torch.arange(64, dtype=torch.float32)
    cache.store(vae, key, expected)
    barrier = threading.Barrier(8)

    def lookup_and_mutate(index):
        barrier.wait()
        value = cache.lookup(vae, key)
        assert value is not None
        original_ptr = value.data_ptr()
        value.add_(index)
        return original_ptr

    with ThreadPoolExecutor(max_workers=8) as pool:
        pointers = list(pool.map(lookup_and_mutate, range(8)))

    assert len(set(pointers)) == 8
    assert torch.equal(cache.lookup(vae, key), expected)


def test_reference_image_cold_then_warm_skips_encode_and_is_exact():
    vae = ImageVAE()
    assets = _image_assets("a" * 64)
    cold = reference.encode_reference_latents_cached(vae, assets)
    warm = reference.encode_reference_latents_cached(vae, assets)

    assert vae.calls == 1
    assert torch.equal(cold.latents[0], warm.latents[0])
    assert cold.latents[0] is not warm.latents[0]


def test_reference_image_hit_mutation_does_not_damage_cache_master():
    vae = ImageVAE()
    assets = _image_assets("a" * 64)
    reference.encode_reference_latents_cached(vae, assets)
    first_hit = reference.encode_reference_latents_cached(vae, assets)
    first_hit.latents[0].add_(99)
    second_hit = reference.encode_reference_latents_cached(vae, assets)

    assert vae.calls == 1
    assert not torch.equal(first_hit.latents[0], second_hit.latents[0])
    assert torch.equal(second_hit.latents[0], torch.zeros_like(second_hit.latents[0]))


def test_reference_image_changed_preprocessed_hash_misses():
    vae = ImageVAE()
    reference.encode_reference_latents_cached(vae, _image_assets("size-a"))
    reference.encode_reference_latents_cached(vae, _image_assets("size-b"))

    assert vae.calls == 2


def test_three_reference_images_cache_independently():
    vae = ImageVAE()
    assets = _image_assets("one", "two", "three")
    events = []
    reference.encode_reference_latents_cached(vae, assets, cache_event=events.append)
    reference.encode_reference_latents_cached(vae, assets, cache_event=events.append)

    assert vae.calls == 3
    assert sum(event.action == "hit" for event in events) == 3
    assert sum(event.action == "miss" for event in events) == 3


def test_reference_video_same_combined_hash_hits_and_preserves_presentation():
    vae = VideoVAE()
    source = _video_source()
    cold = reference_video.encode_reference_video_cached(vae, source)
    warm = reference_video.encode_reference_video_cached(vae, source)

    assert vae.calls == 1
    assert torch.equal(cold.block["latent"], warm.block["latent"])
    assert cold.block["latent"].data_ptr() != warm.block["latent"].data_ptr()
    assert torch.equal(cold.item["data"], warm.item["data"])
    assert cold.item["timestamps"] == warm.item["timestamps"]


def test_reference_video_frame_size_and_source_identity_changes_miss():
    vae = VideoVAE()
    sources = [
        _video_source(target_frames=22),
        _video_source(target_frames=39),
        _video_source(
            target_frames=22,
            size_mode=reference_video.REFERENCE_VIDEO_SIZE_MATCH_OUTPUT,
        ),
        _video_source(target_frames=22, value=0.5),
    ]
    assert len({source.combined_hash for source in sources}) == len(sources)
    for source in sources:
        reference_video.encode_reference_video_cached(vae, source)

    assert vae.calls == len(sources)


def test_reference_video_cache_does_not_retain_source_frames():
    vae = VideoVAE()
    source = _video_source()
    frames_ref = weakref.ref(source.frames)
    assets = reference_video.encode_reference_video_cached(vae, source)
    del assets
    del source
    gc.collect()

    assert frames_ref() is None
    assert inspect_ref_encode_cache()["resident_entries"] == 1


def test_reference_audio_cpu_output_cold_then_warm_is_exact():
    vae = AudioVAE()
    source = _audio_source()
    cold = reference_audio.encode_reference_audio_cached(vae, source)
    warm = reference_audio.encode_reference_audio_cached(vae, source)

    assert vae.calls == 1
    assert torch.equal(cold.audio_latent, warm.audio_latent)
    assert cold.audio_latent.data_ptr() != warm.audio_latent.data_ptr()


def test_reference_audio_source_or_vae_change_misses():
    first_vae = AudioVAE()
    second_vae = AudioVAE()
    reference_audio.encode_reference_audio_cached(first_vae, _audio_source("a"))
    reference_audio.encode_reference_audio_cached(first_vae, _audio_source("b"))
    reference_audio.encode_reference_audio_cached(second_vae, _audio_source("a"))

    assert first_vae.calls == 2
    assert second_vae.calls == 1


def test_reference_audio_non_cpu_output_is_safely_bypassed(monkeypatch):
    vae = AudioVAE()
    source = _audio_source()
    calls = []

    def fake_encode(_vae, active_source):
        calls.append(active_source.combined_hash)
        return ReferenceAudioAssets(
            source=active_source,
            audio_latent=torch.empty((1, 32, 2, 9), device="meta"),
        )

    monkeypatch.setattr(reference_audio, "encode_reference_audio", fake_encode)
    events = []
    first = reference_audio.encode_reference_audio_cached(
        vae, source, cache_event=events.append
    )
    second = reference_audio.encode_reference_audio_cached(
        vae, source, cache_event=events.append
    )

    assert first.audio_latent.device.type == "meta"
    assert second.audio_latent.device.type == "meta"
    assert calls == [source.combined_hash, source.combined_hash]
    assert [event.action for event in events] == [
        "miss",
        "bypass",
        "miss",
        "bypass",
    ]
    assert all(
        event.reason == "payload_not_cpu"
        for event in events
        if event.action == "bypass"
    )


def test_detailed_cache_report_has_kind_totals_and_resident_limits():
    vae = ImageVAE()
    events = []
    assets = _image_assets("a")
    reference.encode_reference_latents_cached(vae, assets, cache_event=events.append)
    reference.encode_reference_latents_cached(vae, assets, cache_event=events.append)
    line = format_ref_encode_cache_diagnostics(events)

    assert line.startswith("Reference encode cache:")
    assert "image hit=1/miss=1/store=1/bypass=0" in line
    assert "total hits=1, misses=1, stores=1, bypasses=0" in line
    assert "MiB/512.0 MiB" in line
    assert "1/16 entries" in line


def test_v38_enables_internal_cache_without_public_schema_change(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_v37_run(_self, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(driving_nodes.H3ContinuumSamplerV37, "run", fake_v37_run)
    result = driving_nodes.H3ContinuumSamplerV38().run(
        aspect="Square 1:1",
        preset="Custom",
        custom_mp=0.3,
        chunks=1,
        chunk_seconds=5.0,
        diagnostics="Off",
    )
    schema = driving_nodes.H3ContinuumSamplerV38.INPUT_TYPES()

    assert result is sentinel
    assert captured["reference_encode_cache"] is True
    assert all(
        "reference_encode_cache" not in group
        for group in (
            schema.get("required", {}),
            schema.get("optional", {}),
            schema.get("hidden", {}),
        )
    )


def test_production_facade_accepts_internal_cache_flag():
    signature = inspect.signature(driving_nodes.H3ContinuumSamplerProduction.run)

    assert "reference_encode_cache" in signature.parameters
    assert signature.parameters["reference_encode_cache"].default is False
    source = inspect.getsource(driving_nodes.H3ContinuumSamplerProduction.run)
    assert "reference_encode_cache=bool(reference_encode_cache)" in source


def test_v37_and_older_do_not_enable_internal_cache(monkeypatch):
    captured = {}

    def fake_v36_run(_self, **kwargs):
        captured.update(kwargs)
        return (), (), {}, "status", None, None

    monkeypatch.setattr(driving_nodes.H3ContinuumSamplerV36, "run", fake_v36_run)
    driving_nodes.H3ContinuumSamplerV37().run(
        guide=None,
        width=32,
        height=32,
    )

    assert "reference_encode_cache" not in captured
    for sampler in (
        driving_nodes.H3ContinuumSamplerV34,
        driving_nodes.H3ContinuumSamplerV35,
        driving_nodes.H3ContinuumSamplerV36,
        driving_nodes.H3ContinuumSamplerV37,
    ):
        schema = sampler.INPUT_TYPES()
        assert "reference_encode_cache" not in schema.get("required", {})
        assert "reference_encode_cache" not in schema.get("optional", {})


def test_cache_flag_is_not_part_of_run_storage_or_public_workflow_identity():
    assert "reference_encode_cache" not in (ROOT / "run_storage.py").read_text(
        encoding="utf-8"
    )
    assert "reference_encode_cache" not in (
        ROOT / "examples" / "workflows" / "MiniMax_H3_Continuum_V38.json"
    ).read_text(encoding="utf-8")


def test_full_reuse_skips_cache_and_partial_regenerate_enters_cached_encode_stage():
    source = (ROOT / "v2" / "sequence.py").read_text(encoding="utf-8")
    partial_regenerate_guard = source.index("if len(preserved)<chunks:")
    image = source.index("encode_reference_latents_cached", partial_regenerate_guard)
    audio = source.index("encode_reference_audio_input", partial_regenerate_guard)
    video = source.index("encode_reference_video_cached", partial_regenerate_guard)
    entries = source.index("entries=preserved[:]", partial_regenerate_guard)

    assert partial_regenerate_guard < image < entries
    assert partial_regenerate_guard < audio < entries
    assert partial_regenerate_guard < video < entries

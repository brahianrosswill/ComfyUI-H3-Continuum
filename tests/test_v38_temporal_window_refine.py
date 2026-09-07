from __future__ import annotations

import copy
import math

import pytest
import torch

from ComfyUI_H3_Continuum_Join.temporal import audio_latent_t, video_latent_t
from ComfyUI_H3_Continuum_Join.v3.refine_target import (
    MODE_AUDIO_ONLY,
    MODE_VIDEO_AUDIO,
    MODE_VIDEO_ONLY,
)
from ComfyUI_H3_Continuum_Join.v3.refine_window import (
    MAGIC,
    MODE_TIME_WINDOW,
    GroupTemporalWindow,
    RefineWindowError,
    resolve_refine_window,
    serializable_window_contract,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass_nodes import (
    H3ContinuumSelectiveSecondPassExperimental,
    REFINE_SCOPE_TIME_WINDOW,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_refine_sampling import (
    _prepare_audio_only_noise_and_mask,
    _prepare_video_audio_noise_and_mask,
    _prepare_video_only_noise_and_mask,
    _restore_outside_temporal_ranges,
    sample_targeted_refine_chunk,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import (
    run_targeted_second_pass_groups,
)


def _physical_group(
    group_id: int,
    logical_chunks: list[int],
    physical_frames: int,
    trim_frames: int,
    *,
    terminal: bool = False,
) -> dict:
    return {
        "group_id": group_id,
        "logical_chunks": logical_chunks,
        "physical_prompt": f"group {group_id + 1}",
        "prompt_policy": "paired_timeline_v1" if terminal else "single",
        "physical_frames": physical_frames,
        "trim_prefix_frames": trim_frames,
        "terminal_merged": terminal,
        "source_width": 128,
        "source_height": 128,
        "source_batch": 1,
        "latent_channels": 24,
        "source_latent_t": video_latent_t(physical_frames),
        "source_latent_h": 8,
        "source_latent_w": 8,
        "source_audio_shape": [1, 32, 2, audio_latent_t(physical_frames)],
    }


def _logical_chunks() -> list[dict]:
    values = ((124, 0), (141, 22), (141, 22))
    cursor = 0
    chunks = []
    for index, (total, trim) in enumerate(values, start=1):
        net = total - trim
        chunks.append(
            {
                "sequence_index": index,
                "chunk_index": index,
                "total_frames": total,
                "trim_frames": trim,
                "net_frames": net,
                "context_frames": trim,
                "expected_video_latent_t": video_latent_t(total),
                "expected_audio_latent_t": audio_latent_t(total),
                "frame_start": cursor,
                "frame_stop": cursor + net,
            }
        )
        cursor += net
    return chunks


def _plan(*, terminal: bool = False, preserve_final_frame: bool = False) -> dict:
    chunks = _logical_chunks()
    if terminal:
        groups = [
            _physical_group(0, [1], 124, 0),
            _physical_group(1, [2, 3], 260, 22, terminal=True),
        ]
        decode_groups = [
            {
                **chunks[0],
                "logical_chunk_indices": [1],
                "terminal_merged": False,
            },
            {
                "total_frames": 260,
                "trim_frames": 22,
                "net_frames": 238,
                "context_frames": 22,
                "expected_video_latent_t": video_latent_t(260),
                "expected_audio_latent_t": audio_latent_t(260),
                "frame_start": 124,
                "frame_stop": 362,
                "logical_chunk_indices": [2, 3],
                "terminal_merged": True,
            },
        ]
    else:
        groups = [
            _physical_group(0, [1], 124, 0),
            _physical_group(1, [2], 141, 22),
            _physical_group(2, [3], 141, 22),
        ]
        decode_groups = [
            {
                **chunk,
                "logical_chunk_indices": [index],
                "terminal_merged": False,
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
    return {
        "magic": "H3_CONTINUUM_ASSEMBLY_PLAN",
        "schema_version": 1,
        "fps": 24,
        "width": 128,
        "height": 128,
        "chunk_seconds": 5.0,
        "target_frames": 360,
        "natural_frames": 362,
        "preserve_final_frame": preserve_final_frame,
        "chunks": chunks,
        "decode_groups": decode_groups,
        "second_pass_contract": {"version": 1, "physical_groups": groups},
    }


def _latents(plan: dict):
    groups = plan["second_pass_contract"]["physical_groups"]
    videos = [
        {
            "samples": torch.full(
                (
                    1,
                    24,
                    group["source_latent_t"],
                    8,
                    8,
                ),
                float(index + 1),
            )
        }
        for index, group in enumerate(groups)
    ]
    audios = [
        {
            "samples": torch.full(
                tuple(group["source_audio_shape"]),
                float(index + 11),
            )
        }
        for index, group in enumerate(groups)
    ]
    return videos, audios


def test_visible_window_resolves_across_groups_without_selecting_prefix():
    plan = _plan()
    before = copy.deepcopy(plan)

    window = resolve_refine_window(5.0, 10.0, plan)

    assert plan == before
    assert window.selected_group_indices == (0, 1)
    assert window.contract["magic"] == MAGIC
    assert window.contract["output_frame_range"] == [120, 240]
    assert window.groups[0].physical_frame_ranges == ((120, 124),)
    assert window.groups[1].physical_frame_ranges == ((22, 138),)
    assert window.groups[1].video_slot_ranges[0][0] >= 7
    assert window.groups[1].audio_tick_ranges[0][0] == audio_latent_t(22)
    assert window.contract["continuation_prefix_policy"] == "always_protected"
    serialized = serializable_window_contract(window)
    serialized["groups"][0]["physical_frame_ranges"].append([0, 1])
    assert window.groups[0].physical_frame_ranges == ((120, 124),)


def test_full_visible_window_clamps_and_excludes_non_visible_tail():
    window = resolve_refine_window(-1.0, 20.0, _plan())

    assert window.contract["output_frame_range"] == [0, 360]
    assert window.selected_group_indices == (0, 1, 2)
    assert window.groups[1].physical_frame_ranges == ((22, 141),)
    assert window.groups[2].physical_frame_ranges == ((22, 139),)


def test_preserved_final_anchor_maps_to_natural_final_frame():
    window = resolve_refine_window(0.0, 15.0, _plan(preserve_final_frame=True))

    assert window.contract["natural_frame_ranges"] == [[0, 359], [361, 362]]
    assert window.groups[2].physical_frame_ranges == ((22, 138), (140, 141))


def test_terminal_logical_half_window_keeps_physical_pair_atomic():
    window = resolve_refine_window(243 / 24, 15.0, _plan(terminal=True))

    assert window.selected_group_indices == (1,)
    group = window.groups[0]
    assert group.logical_chunks == (2, 3)
    assert group.terminal_merged is True
    assert group.physical_frame_ranges == ((141, 258),)


@pytest.mark.parametrize(
    ("start", "end", "match"),
    (
        (5.0, 5.0, "start must be less"),
        (6.0, 5.0, "start must be less"),
        (math.nan, 5.0, "finite"),
        (20.0, 30.0, "does not overlap"),
    ),
)
def test_invalid_or_empty_window_fails_before_execution(start, end, match):
    with pytest.raises(RefineWindowError, match=match):
        resolve_refine_window(start, end, _plan())


def _small_group_window() -> GroupTemporalWindow:
    return GroupTemporalWindow(
        group_index=0,
        logical_chunks=(1,),
        terminal_merged=False,
        visible_frame_ranges=((0, 1),),
        physical_frame_ranges=((0, 1),),
        video_slot_ranges=((2, 4),),
        audio_tick_ranges=((3, 6),),
    )


def _noise(tensor, _seed, _batch_inds):
    return torch.ones_like(tensor)


def test_target_masks_apply_only_selected_video_and_audio_ranges():
    video = torch.zeros(1, 24, 6, 2, 2)
    audio = torch.zeros(1, 32, 2, 8)
    window = _small_group_window()
    nested = lambda values: values

    _noise_value, video_only_mask = _prepare_video_only_noise_and_mask(
        video,
        audio,
        seed=1,
        prepare_noise_fn=_noise,
        nested_builder=nested,
        temporal_window=window,
    )
    assert torch.count_nonzero(video_only_mask[0][:, :, :2]) == 0
    assert torch.all(video_only_mask[0][:, :, 2:4] == 1)
    assert torch.count_nonzero(video_only_mask[0][:, :, 4:]) == 0
    assert torch.count_nonzero(video_only_mask[1]) == 0

    _noise_value, audio_only_mask = _prepare_audio_only_noise_and_mask(
        video,
        audio,
        seed=1,
        prepare_noise_fn=_noise,
        nested_builder=nested,
        temporal_window=window,
    )
    assert torch.count_nonzero(audio_only_mask[0]) == 0
    assert torch.all(audio_only_mask[1][..., 3:6] == 1)
    assert torch.count_nonzero(audio_only_mask[1][..., :3]) == 0
    assert torch.count_nonzero(audio_only_mask[1][..., 6:]) == 0

    _noise_value, both_mask = _prepare_video_audio_noise_and_mask(
        video,
        audio,
        seed=1,
        prepare_noise_fn=_noise,
        nested_builder=nested,
        temporal_window=window,
    )
    assert torch.equal(both_mask[0], video_only_mask[0])
    assert torch.equal(both_mask[1], audio_only_mask[1])


def test_outside_window_restore_is_bit_exact():
    source = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1)
    sampled = torch.full_like(source, 99.0)

    restored = _restore_outside_temporal_ranges(
        sampled,
        source,
        ((2, 4),),
        time_dim=2,
        name="Video",
    )

    assert torch.equal(restored[:, :, :2], source[:, :, :2])
    assert torch.equal(restored[:, :, 4:], source[:, :, 4:])
    assert torch.all(restored[:, :, 2:4] == 99.0)
    assert torch.equal(source, torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1))


def test_target_dispatch_passes_window_without_changing_no_window_path():
    calls = []
    window = _small_group_window()

    def fake(**kwargs):
        calls.append(kwargs)
        return kwargs["latent"]

    base = dict(
        model="model",
        conditioning=[],
        latent={"samples": "samples"},
        sampler="sampler",
        sigmas=torch.tensor([1.0, 0.0]),
        seed=7,
        enable_preview=False,
    )
    sample_targeted_refine_chunk(
        **base,
        refine_target=MODE_VIDEO_ONLY,
        legacy_sample_fn=fake,
    )
    sample_targeted_refine_chunk(
        **base,
        refine_target=MODE_VIDEO_ONLY,
        temporal_window=window,
        video_only_window_sample_fn=fake,
    )

    assert "temporal_window" not in calls[0]
    assert calls[1]["temporal_window"] is window


@pytest.mark.parametrize("target", (MODE_VIDEO_ONLY, MODE_AUDIO_ONLY, MODE_VIDEO_AUDIO))
def test_temporal_window_skips_outside_groups_before_any_work(target):
    plan = _plan()
    videos, audios = _latents(plan)
    calls = {"encode": [], "clone": [], "sample": []}

    def encode(_clip, prompt, **_kwargs):
        calls["encode"].append(prompt)
        return [prompt]

    def clone(model, **kwargs):
        calls["clone"].append(kwargs)
        return model

    def sample(**kwargs):
        calls["sample"].append(kwargs)
        return {
            "video": kwargs["latent"]["video"] + 100.0,
            "audio": kwargs["latent"]["audio"] + 200.0,
        }

    output_videos, output_audios, updated_plan, status = run_targeted_second_pass_groups(
        model="model",
        clip="clip",
        sampler="sampler",
        sigmas=torch.tensor([0.6, 0.3, 0.0]),
        video_latents=videos,
        audio_latents=audios,
        assembly_plan=plan,
        refine_seed=91,
        refine_target=target,
        refine_scope=MODE_TIME_WINDOW,
        window_start_sec=5.0,
        window_end_sec=10.0,
        encode_prompt_fn=encode,
        latent_builder=lambda video, audio: {"video": video, "audio": audio},
        sample_fn=sample,
        stream_extractor=lambda value: (value["video"], value["audio"]),
        clone_model_fn=clone,
        enable_preview=False,
    )

    assert len(calls["encode"]) == len(calls["clone"]) == len(calls["sample"]) == 2
    assert [call["temporal_window"].group_index for call in calls["sample"]] == [0, 1]
    assert output_videos[2] is videos[2]
    assert output_audios[2] is audios[2]
    assert "Refine Window: timeline=final_visible_output" in status
    assert "refine_window_contract" not in updated_plan["second_pass_contract"]


def test_experimental_node_appends_time_window_inputs_without_changing_default():
    required = H3ContinuumSelectiveSecondPassExperimental.INPUT_TYPES()["required"]

    assert REFINE_SCOPE_TIME_WINDOW in required["refine_scope"][0]
    assert required["refine_scope"][1]["default"] == "All"
    assert required["window_start_sec"][1]["default"] == 0.0
    assert required["window_end_sec"][1]["default"] == 5.0

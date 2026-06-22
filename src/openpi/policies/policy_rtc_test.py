"""Unit tests for the RTC obs-field extraction in Policy.infer (M2)."""

from __future__ import annotations

import numpy as np

from openpi.policies.policy import _pop_rtc_kwargs


def test_pop_rtc_kwargs_extracts_and_leaves_obs_intact() -> None:
    prev = np.zeros((24, 14), dtype=np.float32)
    inputs = {
        "image": "IMG",
        "state": "STATE",
        "prompt": "do the thing",
        "prev_action_chunk": prev,
        "inference_delay": 3,
        "execute_horizon": 8,
        "prefix_attention_schedule": "exp",
        "max_guidance_weight": 5.0,
    }
    rtc = _pop_rtc_kwargs(inputs)

    assert rtc["prev_action_chunk"] is prev
    assert rtc["inference_delay"] == 3
    assert rtc["execute_horizon"] == 8
    assert rtc["prefix_attention_schedule"] == "exp"
    assert rtc["max_guidance_weight"] == 5.0
    # Model observation fields are untouched; RTC fields are removed.
    assert set(inputs) == {"image", "state", "prompt"}


def test_pop_rtc_kwargs_off_without_prev_chunk() -> None:
    # No prev_action_chunk -> RTC OFF even if other knobs leak in; nothing popped.
    inputs = {"state": "STATE", "inference_delay": 3}
    assert _pop_rtc_kwargs(inputs) == {}
    assert set(inputs) == {"state", "inference_delay"}


def test_pop_rtc_kwargs_prev_only_minimal() -> None:
    prev = np.ones((4, 14), dtype=np.float32)
    inputs = {"state": "S", "prev_action_chunk": prev}
    rtc = _pop_rtc_kwargs(inputs)
    assert list(rtc) == ["prev_action_chunk"]
    assert "prev_action_chunk" not in inputs

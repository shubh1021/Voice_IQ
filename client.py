# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Voiceiq Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AudioAction, AudioObservation


class VoiceIQEnv(
    EnvClient[AudioAction, AudioObservation, State]
):
    """Client for the VoiceIQ Environment."""

    def _step_payload(self, action: AudioAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[AudioObservation]:
        obs_data = payload.get("observation", {})
        observation = AudioObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
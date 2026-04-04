# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the VoiceIQ Environment."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.") from e

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AudioAction, AudioObservation
from server.voiceiq_environment import VoiceIQEnvironment


# Create the base app
app = create_app(
    VoiceIQEnvironment,
    AudioAction,
    AudioObservation,
    env_name="voiceiq",
    max_concurrent_envs=1,
)


# --- 3 REQUIRED EXTRA ENDPOINTS ---

@app.get("/tasks")
def get_tasks():
    """Returns the 3 tasks this environment supports."""
    return JSONResponse({
        "tasks": [
            {
                "task_id": "single_emotion",
                "difficulty": "easy",
                "description": "Identify a single strong emotion from a customer audio clip.",
                "scoring": "Exact tone match + intensity in range 0.7-1.0"
            },
            {
                "task_id": "low_intensity",
                "difficulty": "medium",
                "description": "Detect subtle or low-intensity emotions that are harder to identify.",
                "scoring": "Exact tone match + intensity in range 0.3-0.7"
            },
            {
                "task_id": "escalation",
                "difficulty": "hard",
                "description": "Detect passive-aggressive tone where words sound polite but intent is hostile. Determine if escalation is needed.",
                "scoring": "Tone match + text_audio_match=False + correct escalation decision"
            }
        ]
    })


@app.get("/grader")
def get_grader():
    """Returns the grader criteria and weights."""
    return JSONResponse({
        "grader_type": "hybrid",
        "structural_weight": 0.8,
        "llm_judge_weight": 0.2,
        "criteria": {
            "tone_correct": 0.138,
            "intensity_in_range": 0.108,
            "escalation_logic": 0.123,
            "escalation_tier_correct": 0.077,
            "text_audio_match_correct": 0.108,
            "pitch_level_valid": 0.077,
            "speaking_pace_valid": 0.077,
            "energy_level_valid": 0.092,
        },
        "thresholds": {
            "pitch": {"low": "<100Hz", "normal": "100-180Hz", "high": ">180Hz"},
            "pace": {"slow": "<110wpm", "normal": "110-160wpm", "fast": ">160wpm"},
            "energy": {"low": "<0.02", "normal": "0.02-0.06", "high": ">0.06"}
        }
    })


@app.get("/baseline")
def get_baseline():
    """Returns baseline scoring info."""
    return JSONResponse({
        "baseline_model": "llama-3.3-70b-versatile",
        "baseline_scores": {
            "single_emotion": {"avg_reward": 0.72, "avg_structural": 0.68, "avg_llm": 0.85},
            "low_intensity": {"avg_reward": 0.58, "avg_structural": 0.54, "avg_llm": 0.72},
            "escalation": {"avg_reward": 0.41, "avg_structural": 0.38, "avg_llm": 0.55},
        },
        "note": "Run inference.py to reproduce these scores."
    })

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
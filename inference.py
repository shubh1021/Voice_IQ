"""
VoiceIQ Inference Script
Runs an LLM agent against the VoiceIQ audio tone analysis environment.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from voiceiq.client import VoiceIQEnv
from voiceiq.models import AudioAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or HF_TOKEN
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or "voiceiq-env:latest"
TASK_NAME = os.getenv("VOICEIQ_TASK", "single_emotion")
BENCHMARK = "voiceiq"
MAX_STEPS = 1  # one clip per episode
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert customer support supervisor analysing audio call features.
    You will receive preprocessed audio features from a customer support call.
    Your job is to analyse these features and return a structured JSON response.

    You must respond with ONLY a valid JSON object with these exact fields:
    {
        "tone": "<one of: angry, frustrated, neutral, happy, sad, passive_aggressive>",
        "intensity": <float 0.0 to 1.0>,
        "pitch_level": "<one of: low, normal, high>",
        "speaking_pace": "<one of: slow, normal, fast>",
        "energy_level": "<one of: low, normal, high>",
        "escalate": <true or false>,
        "escalation_tier": "<one of: none, senior_agent, manager, emergency>",
        "text_audio_match": <true or false>,
        "reasoning": "<brief explanation of your analysis>"
    }

    Guidelines:
    - Male pitch: 85-155Hz, Female pitch: 180-220Hz. Above range = stressed/angry.
    - WPM above 160 = fast/agitated. Below 110 = slow/suppressed.
    - RMS energy above 0.06 = loud/angry. Below 0.02 = quiet/sad.
    - passive_aggressive = polite words but hostile tone (text_audio_match = false)
    - escalate = true if tone is angry or passive_aggressive AND intensity >= 0.6
    - escalation_tier: none < senior_agent (intensity 0.6-0.79) < manager (0.8-0.89) < emergency (0.9+)
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs) -> str:
    return textwrap.dedent(f"""
        Analyse this customer support audio clip:

        Clip ID: {obs.clip_id}
        Task: {obs.task_id}
        Duration: {obs.duration_seconds:.1f}s
        Transcript: "{obs.transcript}"

        Audio Features:
        - Mean Pitch: {obs.mean_pitch_hz:.1f} Hz
        - Pitch Variance: {obs.pitch_variance:.2f}
        - Pitch Slope: {obs.pitch_slope:.4f}
        - Pitch Range: {obs.pitch_range:.1f} Hz
        - RMS Energy: {obs.rms_energy:.4f}
        - Energy Variance: {obs.energy_variance:.6f}
        - Energy Trend: {obs.energy_trend}
        - Words Per Minute: {obs.words_per_minute:.1f}
        - Silence Ratio: {obs.silence_ratio:.2f}
        - Pause Count: {obs.pause_count}
        - Speech Rate Change: {obs.speech_rate_change}
        - Negative Word Count: {obs.negative_word_count}
        - Question Count: {obs.question_count}
        - Filler Word Count: {obs.filler_word_count}

        Respond with ONLY the JSON object.
    """).strip()


def get_agent_action(client: OpenAI, obs) -> AudioAction:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=300,
            stream=False,
        )
        import json
        text = completion.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return AudioAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Return safe fallback
        return AudioAction(
            tone="neutral",
            intensity=0.5,
            pitch_level="normal",
            speaking_pace="normal",
            energy_level="normal",
            escalate=False,
            escalation_tier="none",
            text_audio_match=True,
            reasoning="Fallback due to parsing error."
        )


async def run_episode(client: OpenAI, env: VoiceIQEnv, task_id: str) -> float:
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            action = get_agent_action(client, obs)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"tone={action.tone},intensity={action.intensity:.2f},escalate={action.escalate}",
                reward=reward,
                done=done,
                error=None
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken+1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["single_emotion", "low_intensity", "escalation"]
    all_scores = []

    for task_id in tasks:
        try:
            if IMAGE_NAME and IMAGE_NAME != "voiceiq-env:latest":
                env = await VoiceIQEnv.from_docker_image(IMAGE_NAME)
            else:
                env = VoiceIQEnv(base_url="https://shubh0107-voiceiq.hf.space")
            score = await run_episode(client, env, task_id)
            all_scores.append(score)
        except Exception as e:
            print(f"[DEBUG] Episode failed for {task_id}: {e}", flush=True)
            all_scores.append(0.0)

    print(f"\n[SUMMARY] avg_score={sum(all_scores)/len(all_scores):.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
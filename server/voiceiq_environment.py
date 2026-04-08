import os
import random
from uuid import uuid4

import librosa
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import AudioAction, AudioObservation
    from clips_dataset import CLIPS
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models import AudioAction, AudioObservation
    from clips_dataset import CLIPS


class VoiceIQEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._clips_dir = os.path.join(os.path.dirname(__file__), "audio_clips")
        self._groq_client = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_clip = None
        self._current_task = None

    @property
    def _client(self):
        if self._groq_client is None:
            from openai import OpenAI as _OpenAI
            self._groq_client = _OpenAI(
                api_key=os.environ.get("API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("HF_TOKEN"),
                base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
            )
        return self._groq_client

    def reset(self, task_id: str = None) -> AudioObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if task_id:
            available = {k: v for k, v in CLIPS.items() if v["task_id"] == task_id}
        else:
            available = CLIPS

        clip_id = random.choice(list(available.keys()))
        self._current_clip = CLIPS[clip_id]
        self._current_task = self._current_clip["task_id"]

        audio_path = os.path.join(self._clips_dir, self._current_clip["file"])
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0_clean = f0[~np.isnan(f0)]
        mean_pitch = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        pitch_variance = float(np.var(f0_clean)) if len(f0_clean) > 0 else 0.0
        pitch_slope = float(np.polyfit(np.arange(len(f0_clean)), f0_clean, 1)[0]) if len(f0_clean) > 1 else 0.0
        pitch_range = float(np.max(f0_clean) - np.min(f0_clean)) if len(f0_clean) > 0 else 0.0

        rms = librosa.feature.rms(y=y)[0]
        rms_energy = float(np.mean(rms))
        energy_variance = float(np.var(rms))
        if np.polyfit(np.arange(len(rms)), rms, 1)[0] > 0.0001:
            energy_trend = "rising"
        elif np.polyfit(np.arange(len(rms)), rms, 1)[0] < -0.0001:
            energy_trend = "falling"
        else:
            energy_trend = "stable"

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = [float(x) for x in np.mean(mfcc, axis=1)]
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        transcript_data = self._client.audio.transcriptions.create(
            model=os.environ.get("WHISPER_MODEL", "whisper-large-v3"),
            file=open(audio_path, "rb"),
            response_format="verbose_json"
        )
        transcript = transcript_data.text
        words = transcript.split()
        word_count = len(words)
        wpm = (word_count / duration) * 60 if duration > 0 else 0.0

        filler_words = ["uh", "um", "like", "you know", "basically", "literally"]
        filler_count = sum(transcript.lower().count(f) for f in filler_words)
        question_count = transcript.count("?")
        negative_words = ["not", "never", "no", "won't", "can't", "didn't", "don't", "horrible", "terrible", "worst", "useless"]
        negative_count = sum(transcript.lower().count(w) for w in negative_words)

        intervals = librosa.effects.split(y, top_db=20)
        speech_samples = sum(end - start for start, end in intervals)
        silence_ratio = 1.0 - (speech_samples / len(y)) if len(y) > 0 else 0.0
        pause_count = max(0, len(intervals) - 1)

        mid = len(rms) // 2
        first_half_energy = np.mean(rms[:mid])
        second_half_energy = np.mean(rms[mid:])
        if second_half_energy > first_half_energy * 1.1:
            speech_rate_change = "speeding_up"
        elif second_half_energy < first_half_energy * 0.9:
            speech_rate_change = "slowing_down"
        else:
            speech_rate_change = "stable"

        self._last_pitch_hz = mean_pitch
        self._last_wpm = wpm
        self._last_rms = rms_energy
        self._last_transcript = transcript
        self._ground_truth = self._current_clip

        return AudioObservation(
            clip_id=clip_id,
            task_id=self._current_task,
            duration_seconds=duration,
            mean_pitch_hz=mean_pitch,
            pitch_variance=pitch_variance,
            pitch_slope=pitch_slope,
            pitch_range=pitch_range,
            rms_energy=rms_energy,
            energy_variance=energy_variance,
            energy_trend=energy_trend,
            words_per_minute=wpm,
            silence_ratio=silence_ratio,
            pause_count=pause_count,
            speech_rate_change=speech_rate_change,
            mfcc_mean=mfcc_mean,
            spectral_centroid=spectral_centroid,
            transcript=transcript,
            word_count=word_count,
            filler_word_count=filler_count,
            question_count=question_count,
            negative_word_count=negative_count,
            avg_word_confidence=1.0,
            ground_truth_tone=None,
            ground_truth_intensity_min=None,
            ground_truth_intensity_max=None,
            done=False,
        )

    def step(self, action: AudioAction) -> AudioObservation:
        self._state.step_count += 1
        clip = self._ground_truth

        feedback = {}
        structural_score = 0.0

        if action.tone.lower() == clip["ground_truth_tone"].lower():
            structural_score += 0.138
            feedback["tone_correct"] = True
        else:
            feedback["tone_correct"] = False

        if clip["intensity_min"] <= action.intensity <= clip["intensity_max"]:
            structural_score += 0.108
            feedback["intensity_in_range"] = True
        else:
            feedback["intensity_in_range"] = False

        if action.tone.lower() in ["angry", "passive_aggressive"]:
            if action.intensity >= 0.6:
                correct = action.escalate == True
            else:
                correct = True
        else:
            correct = True
        if correct:
            structural_score += 0.123
        feedback["escalation_logic"] = correct

        if not action.escalate:
            tier_correct = action.escalation_tier == "none"
        elif action.intensity >= 0.8:
            tier_correct = action.escalation_tier in ["manager", "emergency"]
        elif action.intensity >= 0.6:
            tier_correct = action.escalation_tier == "senior_agent"
        else:
            tier_correct = action.escalation_tier == "none"
        if tier_correct:
            structural_score += 0.077
        feedback["escalation_tier_correct"] = tier_correct

        if action.text_audio_match == clip["text_audio_match"]:
            structural_score += 0.108
            feedback["text_audio_match_correct"] = True
        else:
            feedback["text_audio_match_correct"] = False

        if self._last_pitch_hz < 100:
            expected_pitch = "low"
        elif self._last_pitch_hz <= 180:
            expected_pitch = "normal"
        else:
            expected_pitch = "high"
        if action.pitch_level.lower() == expected_pitch:
            structural_score += 0.077
        feedback["pitch_level_valid"] = action.pitch_level.lower() == expected_pitch

        if self._last_wpm < 110:
            expected_pace = "slow"
        elif self._last_wpm <= 160:
            expected_pace = "normal"
        else:
            expected_pace = "fast"
        if action.speaking_pace.lower() == expected_pace:
            structural_score += 0.077
        feedback["speaking_pace_valid"] = action.speaking_pace.lower() == expected_pace

        if self._last_rms < 0.02:
            expected_energy = "low"
        elif self._last_rms <= 0.06:
            expected_energy = "normal"
        else:
            expected_energy = "high"
        if action.energy_level.lower() == expected_energy:
            structural_score += 0.092
        feedback["energy_level_valid"] = action.energy_level.lower() == expected_energy

        llm_score = 0.0
        if action.reasoning:
            try:
                judge_prompt = f"""You are evaluating an AI agent's audio tone analysis.

Audio transcript: "{self._last_transcript}"
Agent detected tone: {action.tone}
Agent's reasoning: {action.reasoning}
Correct tone: {clip["ground_truth_tone"]}

Score the reasoning from 0.0 to 1.0 based on:
- Does the reasoning correctly explain the detected tone?
- Is the reasoning consistent with the transcript?
- Is the logic sound?

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""

                response = self._client.chat.completions.create(
                    model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                llm_score = float(response.choices[0].message.content.strip())
                llm_score = max(0.0, min(1.0, llm_score))
            except Exception:
                llm_score = 0.5

        reward = (structural_score * 0.8) + (llm_score * 0.2)

        return AudioObservation(
            clip_id=self._current_clip["file"],
            task_id=self._current_task,
            duration_seconds=0.0,
            mean_pitch_hz=self._last_pitch_hz,
            pitch_variance=0.0,
            pitch_slope=0.0,
            pitch_range=0.0,
            rms_energy=self._last_rms,
            energy_variance=0.0,
            energy_trend="stable",
            words_per_minute=self._last_wpm,
            silence_ratio=0.0,
            pause_count=0,
            speech_rate_change="stable",
            mfcc_mean=[0.0] * 13,
            spectral_centroid=0.0,
            transcript=self._last_transcript,
            word_count=0,
            filler_word_count=0,
            question_count=0,
            negative_word_count=0,
            avg_word_confidence=1.0,
            structural_score=structural_score,
            llm_score=llm_score,
            reward=reward,
            grader_feedback=feedback,
            done=True,
        )

    @property
    def state(self) -> State:
        return self._state
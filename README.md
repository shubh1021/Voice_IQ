# VoiceIQ — Audio Tone Analysis Environment

I've been researching affective computing for a while now, and one thing that always 
bothered me is how most emotion detection systems only look at text. But human emotion 
lives in the voice — the way someone says "fine" tells you everything the word doesn't.

That's what VoiceIQ is about.

## What it does

VoiceIQ is an OpenEnv environment that trains AI agents to analyse customer support 
audio calls and make real decisions — detect the caller's emotional tone, judge the 
intensity, and decide whether to escalate the call and to whom.

The environment processes raw audio using librosa for acoustic feature extraction and 
Whisper for transcription, then presents the agent with a rich set of features to 
reason over.

## The three tasks

**Task 1 — Single Emotion (Easy)**  
The caller is clearly angry, happy, or sad. Strong acoustic signal, obvious tone. 
Agents that can't handle this one have no business being in a support centre.

**Task 2 — Low Intensity (Medium)**  
Same emotions, but subtle. The difference between someone who is mildly frustrated 
and someone who is neutral is genuinely hard — for humans too.

**Task 3 — Passive Aggressive Detection (Hard)**  
This is the one I'm most proud of. The caller says something like *"No worries, I'll 
just wait another week I guess"* — polite words, hostile intent. The agent has to 
detect that the text and audio don't match, flag it as passive-aggressive, and make 
the right escalation call.

No public dataset has passive-aggressive labels. I built these examples myself because 
the research gap is real and the problem is genuinely hard even for frontier models.

## How the grader works

The grader is hybrid — 80% deterministic, 20% LLM judge.

The deterministic part checks things like: did the agent get the tone right, is the 
intensity in the acceptable range, does the escalation decision follow from the 
intensity, does the tier match. These checks are fully reproducible.

The LLM judge reads the agent's reasoning and scores whether it actually makes sense 
given the transcript. This rewards agents that understand *why* a tone is what it is, 
not just ones that pattern match.

| Check | Weight |
|---|---|
| Tone correct | 13.8% |
| Intensity in range | 10.8% |
| Escalation logic | 12.3% |
| Escalation tier | 7.7% |
| Text-audio match | 10.8% |
| Pitch level | 7.7% |
| Speaking pace | 7.7% |
| Energy level | 9.2% |
| LLM judge | 20.0% |

## Audio features extracted

Every reset() call runs the full pipeline on the selected clip:

- **Pitch** — mean Hz, variance, slope, range
- **Energy** — RMS, variance, trend (rising/falling/stable)
- **Speaking pace** — words per minute, silence ratio, pause count, rate change
- **Timbre** — 13 MFCCs, spectral centroid
- **Whisper transcription** — transcript, word count, negative words, fillers, questions

## Connecting to this environment
```python
from voiceiq import AudioAction, VoiceIQEnv
import asyncio

async def main():
    async with VoiceIQEnv.from_env("shubh0107/voiceiq") as env:
        result = await env.reset()
        obs = result.observation
        
        print(f"Transcript: {obs.transcript}")
        print(f"Mean pitch: {obs.mean_pitch_hz:.1f} Hz")
        
        action = AudioAction(
            tone="angry",
            intensity=0.85,
            pitch_level="high",
            speaking_pace="fast",
            energy_level="high",
            escalate=True,
            escalation_tier="manager",
            text_audio_match=True,
            reasoning="High pitch variance and fast pace indicate strong anger."
        )
        
        result = await env.step(action)
        print(f"Reward: {result.reward:.3f}")

asyncio.run(main())
```

## Running the baseline
```bash
export GROQ_API_KEY=your_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export LOCAL_IMAGE_NAME=openenv-voiceiq:latest

python inference.py
```

Expected output:
[START] task=single_emotion env=voiceiq model=llama-3.3-70b-versatile
[STEP] step=1 action=tone=angry,intensity=0.85,escalate=True reward=0.71 done=true error=null
[END] success=true steps=1 score=0.714 rewards=0.71

## Dataset

- **Tasks 1 & 2** — RAVDESS (Ryerson Audio-Visual Database of Emotional Speech), 
  a validated academic dataset with 8 emotion categories and two intensity levels
- **Task 3** — Handcrafted passive-aggressive scripts generated with gTTS, because 
  no public dataset covers this specific pattern well enough

## What I learned building this

This was my first time using Docker seriously, and integrating it with HF Spaces and 
OpenEnv was a real learning curve. Getting the import paths to resolve correctly 
between local development and the Docker container alone took longer than I'd like 
to admit.

But the part that actually challenged me was Task 3. Defining what passive-aggressive 
*sounds like* acoustically — lower energy than angry, deliberate pace, falling pitch 
at the end of sentences — and then building a grader that can check for those signals 
deterministically, that was the interesting problem.

## HF Space

[shubh0107/voiceiq](https://huggingface.co/spaces/shubh0107/voiceiq)
from typing import Optional, List
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class AudioObservation(Observation):
    """What the environment sends to the agent — preprocessed audio features."""
    # --- Identity ---
    clip_id: str = Field(..., description="It can be a unique identifier for the audio clip being analyzed. This ID can be used to track and reference specific audio samples throughout the analysis process.")
    task_id: str = Field(..., description="It can be a unique identifier for the task to which the audio clip belongs. This ID can be used to track and reference specific tasks throughout the analysis process.")
    duration_seconds: float = Field(..., description="The duration of the audio clip in seconds. This can be used to calculate speaking rates and other time-based features.")
    # --- Pitch ---
    mean_pitch_hz: float = Field(..., description="Average pitch of the speaker in Hz. Adult female frequency: 180–220Hz and adult male: 85–155Hz. if the range is above average frequency it may indicate excitement, nervousness, anger or scared, while a lower range may suggest calmness or sadness.")
    pitch_variance: float = Field(..., description="Variance in pitch throughout the audio. A high variance may indicate emotional expressiveness, while a low variance may suggest monotony or lack of emotional engagement.")
    pitch_slope: float = Field(..., description="""Trend of pitch over time. An upward slope may indicate increasing excitement or engagement, while a downward slope may suggest decreasing energy or interest.Falling pitch at end = passive-aggressive "fine, whatever" pattern.""")
    pitch_range: Optional[float] = Field(default=None, description="Difference between the maximum and minimum pitch. A wider range may indicate greater emotional expressiveness, while a narrower range may suggest monotony or lack of emotional engagement. Wide range = expressive/emotional. Narrow = controlled or suppressed.")


    # --- Energy ---
    rms_energy: float = Field(..., description="Root mean square energy of the audio signal. Higher energy levels may indicate excitement, anger, or enthusiasm, while lower energy levels may suggest calmness, sadness, or fatigue.")
    energy_variance: float = Field(..., description="Variance in energy throughout the audio. A high variance may indicate emotional expressiveness, while a low variance may suggest monotony or lack of emotional engagement.")
    energy_trend: str = Field(..., description="Trend of energy over time. An upward trend may indicate increasing excitement or engagement, while a downward trend may suggest decreasing energy or interest.")

    # --- Speaking Rate ---
    words_per_minute: float = Field(..., description="Average number of words spoken per minute. A higher speaking rate may indicate excitement, nervousness, or urgency, while a lower speaking rate may suggest calmness, sadness, or fatigue.")
    silence_ratio: float = Field(..., description="Ratio of silent segments to total audio duration. A higher silence ratio may indicate nervousness, hesitation, or uncertainty, while a lower silence ratio may suggest confidence and fluency.")
    pause_count: int = Field(..., description="Number of pauses detected in the audio. A higher pause count may indicate nervousness, hesitation, or uncertainty, while a lower pause count may suggest confidence and fluency.")
    speech_rate_change: str = Field(..., description="Change in speaking rate over time. An increasing speaking rate may indicate growing excitement or urgency, while a decreasing speaking rate may suggest calming down or losing interest.")

    # --- Timbre ---
    mfcc_mean: List[float] = Field(..., description="Mean of the Mel-frequency cepstral coefficients. Represents the timbral characteristics of the audio. Changes in MFCCs can indicate changes in vocal quality, emotional state, or even health conditions.")
    spectral_centroid: float = Field(..., description="Centroid of the frequency spectrum. Indicates the 'brightness' of the audio. Higher values may indicate a brighter sound, while lower values suggest a darker sound.")

    # --- Whisper Text ---
    transcript: str = Field(..., description="Transcribed text from the audio using a speech-to-text model. Provides the content of what was said, which can be analyzed for sentiment, keywords, and other linguistic features.")
    word_count: int = Field(..., description="Number of words in the transcript. A higher word count may indicate a more verbose speaker, while a lower word count may suggest brevity or conciseness. ")
    filler_word_count: int = Field(..., description="Number of filler words (e.g., 'um', 'uh') in the transcript. A higher filler word count may indicate nervousness, hesitation, or uncertainty, while a lower filler word count may suggest confidence and fluency.")
    question_count: int = Field(..., description="Number of questions in the transcript. A higher question count may indicate engagement, curiosity, or uncertainty, while a lower question count may suggest a more declarative speaking style.")
    negative_word_count: int = Field(..., description="Number of negative words in the transcript. A higher negative word count may indicate a more negative sentiment or emotional state, while a lower negative word count may suggest a more positive or neutral sentiment.")
    avg_word_confidence: float = Field(..., description="Average confidence score for each word in the transcript. Higher confidence scores indicate more accurate transcriptions, while lower scores may suggest uncertainty or potential errors in the transcription process.")

    # --- Grader fields (hidden from agent, used server-side) ---
    ground_truth_tone: Optional[str] = Field(default=None, description="The label given in the dataset for the tone of the audio. Used for grading the agent's predictions.")
    ground_truth_intensity_min: Optional[float] = Field(default=None, description= "The minimum intensity value in the dataset for the audio. Used for grading the agent's predictions.")
    ground_truth_intensity_max: Optional[float] = Field(default=None, description="The maximum intensity value in the dataset for the audio. Used for grading the agent's predictions.")

    # --- Scores (returned after step()) ---
    structural_score: Optional[float] = Field(default=None, description="A score evaluating the structural accuracy of the agent's predictions compared to the ground truth. This could be based on how well the predicted tone, intensity, and other features match the expected values.")
    llm_score: Optional[float] = Field(default=None, description="A score evaluating the quality of the agent's reasoning and explanations, if provided. This could be based on the coherence, relevance, and depth of the agent's reasoning compared to the ground truth or expected reasoning.")
    reward: Optional[float] = Field(default=None, description="A combined reward score that takes into account both the structural accuracy and the quality of reasoning. This reward can be used to guide the agent's learning and improvement over time.")
    grader_feedback: Optional[dict] = Field(default=None, description="Feedback from the grader regarding the agent's performance.")
    done: bool = Field(default=False, description="Indicates whether the audio analysis is complete.")


class AudioAction(Action):
    """What the agent sends back after analysing the audio features."""

    tone: str = Field(..., description="It can be one of the following: neutral, happy, sad, angry, fearful, disgusted, surprised. The tone of the speaker can provide insights into their emotional state and intentions. For example, a happy tone may indicate positive emotions and engagement, while an angry tone may suggest frustration or hostility.")
    intensity: float = Field(..., description="The intensity of the speaker's voice. This can be a measure of how loud or soft the voice is. Higher intensity may indicate excitement, anger, or enthusiasm, while lower intensity may suggest calmness, sadness, or fatigue.")
    pitch_level: str = Field(..., description="The pitch level of the speaker's voice. This can be categorized as from medium to normal. A higher pitch level may indicate excitement, nervousness, or urgency, while a lower pitch level may suggest calmness, sadness, or fatigue.")
    speaking_pace: str = Field(..., description="The pace at which the speaker is talking. This can be slow, normal, or fast. A faster speaking pace may indicate excitement, nervousness, or urgency, while a slower speaking pace may suggest calmness, sadness, or fatigue.")
    energy_level: str = Field(..., description="The energy level of the speaker's voice. This can be categorized as low, medium, or high. Higher energy levels may indicate excitement, anger, or enthusiasm, while lower energy levels may suggest calmness, sadness, or fatigue.")
    escalate: bool = Field(..., description="Indicates whether the speaker wants to escalate the issue. This can be inferred from the audio features and the content of the speech. If the speaker is showing signs of frustration, anger, or urgency, they may be more likely to want to escalate the issue to a supervisor or higher authority.")
    escalation_tier: str = Field(..., description="If the speaker wants to escalate, this field indicates the tier of escalation (none, senior_agent, manager, emergency). The escalation tier can be determined based on the severity of the speaker's emotional state and the urgency of the situation. For example, a speaker showing signs of extreme frustration or anger may require a higher tier of escalation.")
    text_audio_match: bool = Field(..., description="Indicates whether the text content of the speech matches the audio features. This can be used to evaluate the consistency of the speaker's communication. ")
    reasoning: Optional[str] = Field(default=None, description="The agent's reasoning behind its predictions. This can provide insights into how the agent is interpreting the audio features and making its decisions. For example, the agent might explain that it predicted an angry tone because of the high energy level, fast speaking pace, and certain keywords in the transcript.")
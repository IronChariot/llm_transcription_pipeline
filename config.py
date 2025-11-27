"""
Configuration for the audio transcription pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Chunking parameters
MIN_CHUNK_DURATION_MIN = 10  # Minimum chunk duration in minutes
MAX_CHUNK_DURATION_MIN = 20  # Maximum chunk duration in minutes
MIN_FINAL_CHUNK_MIN = 5      # Minimum duration for the final chunk
OVERLAP_RATIO = 0.10         # 10% overlap between chunks

# Output settings
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Transcription prompt for speaker diarization
TRANSCRIPTION_PROMPT = """Transcribe this audio recording with speaker diarization.

Instructions:
1. Label speakers as **Speaker 1:**, **Speaker 2:**, etc. (bold format, with colon)
2. Transcribe all spoken words accurately
3. Use a new line for each speaker change
4. Include any significant pauses or interruptions in brackets, e.g., [pause], [crosstalk]

Format each line as:
**Speaker N:** [their spoken words]

After the complete transcript, append a JSON block describing each speaker. Use this exact format:

```json
[
  {
    "label": "Speaker 1",
    "voice_description": "male, irish accent, deep baritone",
    "role_estimation": "senior consultant, project manager, EngineerCo staff"
  },
  {
    "label": "Speaker 2",
    "voice_description": "female, american accent, mid-range",
    "role_estimation": "client representative, UK government staff"
  }
]
```

Include voice characteristics (gender, accent, pitch/tone) and any role/affiliation you can infer from the conversation.

Begin transcription:"""

# Prompt for reconciling speakers across chunks
RECONCILIATION_PROMPT = """You are given speaker metadata from multiple chunks of the same audio recording. Each chunk was transcribed separately, so the same person may have different "Speaker N" labels in different chunks.

Your task is to match speakers across chunks based on their voice descriptions and roles, then assign each unique person a consistent "Person N" label.

Here is the speaker metadata from each chunk:

{chunk_metadata}

Analyze the voice descriptions and role estimations to identify which speakers across different chunks are the same person. Then output a JSON mapping in this exact format:

```json
[
  {{
    "master_label": "Person 1",
    "voice_description": "combined description of this person's voice",
    "role_estimation": "best estimate of their role",
    "chunk_labels": [
      {{"chunk": 1, "label": "Speaker 1"}},
      {{"chunk": 2, "label": "Speaker 1"}},
      {{"chunk": 3, "label": "Speaker 3"}}
    ]
  }},
  {{
    "master_label": "Person 2",
    "voice_description": "combined description of this person's voice",
    "role_estimation": "best estimate of their role",
    "chunk_labels": [
      {{"chunk": 1, "label": "Speaker 2"}},
      {{"chunk": 3, "label": "Speaker 1"}}
    ]
  }}
]
```

Important:
- Every speaker from every chunk must appear exactly once in the mapping
- Use "Person 1", "Person 2", etc. as master labels
- Include all chunks where each person speaks in their chunk_labels array
- If uncertain whether two speakers are the same person, keep them separate

Output only the JSON block, no other text."""

# Model settings
GEMINI_MODEL = "gemini-3-pro-preview"


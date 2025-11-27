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
1. Identify different speakers and label them consistently (e.g., "Speaker 1:", "Speaker 2:", etc.)
2. Transcribe all spoken words accurately
3. Use a new line for each speaker change
4. Include any significant pauses or interruptions in brackets, e.g., [pause], [crosstalk]
5. If a speaker's identity is clear from context (e.g., they introduce themselves), you may use their name instead of "Speaker N"

Format each line as:
Speaker N: [their spoken words]

Begin transcription:"""

# Model settings
GEMINI_MODEL = "gemini-3-pro-preview"


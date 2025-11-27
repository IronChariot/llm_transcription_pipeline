"""
Google Gemini transcription provider.
"""
import os
from typing import Optional

from google import genai
from google.genai import types

from .base import TranscriptionProvider, TranscriptionError
from config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiProvider(TranscriptionProvider):
    """
    Transcription provider using Google Gemini API.
    
    Uses the google.genai library to upload audio files and
    request transcription with speaker diarization.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY from config.
            model: Model name to use. If None, uses GEMINI_MODEL from config.
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "Gemini"
    
    def transcribe(self, audio_path: str, prompt: str) -> str:
        """
        Transcribe an audio file using Google Gemini.
        
        Args:
            audio_path: Path to the audio file to transcribe
            prompt: Instruction prompt for the transcription
            
        Returns:
            The transcribed text with speaker labels
        """
        self.validate_audio_file(audio_path)
        
        try:
            # Upload the audio file
            print(f"  Uploading {os.path.basename(audio_path)} to Gemini...")
            
            # Determine MIME type based on file extension
            ext = os.path.splitext(audio_path)[1].lower()
            mime_types = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.webm': 'audio/webm',
            }
            mime_type = mime_types.get(ext, 'audio/mpeg')
            
            # Upload the file
            uploaded_file = self.client.files.upload(
                file=audio_path,
                config=types.UploadFileConfig(mime_type=mime_type)
            )
            
            print(f"  Requesting transcription...")
            
            # Generate transcription
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=mime_type
                            ),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ]
            )
            
            # Extract the transcription text
            if response.text:
                return response.text.strip()
            else:
                raise TranscriptionError(
                    "Empty response from Gemini API",
                    self.name
                )
                
        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(
                f"Transcription failed: {str(e)}",
                self.name,
                original_error=e
            )


if __name__ == "__main__":
    # Test the provider
    import sys
    from config import TRANSCRIPTION_PROMPT
    
    if len(sys.argv) < 2:
        print("Usage: python -m providers.gemini <audio_file>")
        sys.exit(1)
    
    provider = GeminiProvider()
    result = provider.transcribe(sys.argv[1], TRANSCRIPTION_PROMPT)
    print("\n--- Transcription ---")
    print(result)


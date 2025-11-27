"""
Abstract base class for transcription providers.
"""
from abc import ABC, abstractmethod
from typing import Optional


class TranscriptionProvider(ABC):
    """
    Abstract base class for audio transcription providers.
    
    Subclasses must implement the transcribe method to handle
    audio-to-text conversion using their specific API.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, prompt: str) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file to transcribe
            prompt: Instruction prompt for the transcription
            
        Returns:
            The transcribed text with speaker labels
            
        Raises:
            TranscriptionError: If transcription fails
        """
        pass
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate that the audio file exists and is accessible.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            True if the file is valid
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        import os
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check for supported extensions
        supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}
        ext = os.path.splitext(audio_path)[1].lower()
        
        if ext not in supported_extensions:
            raise ValueError(f"Unsupported audio format: {ext}. Supported: {supported_extensions}")
        
        return True


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""
    
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


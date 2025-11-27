"""
Audio extraction module for extracting audio from video files.
"""
import os
import tempfile
from typing import Optional

from pydub import AudioSegment


# Video extensions that we can extract audio from
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}

# Audio extensions that don't need extraction
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a video file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a video file
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def is_audio_file(file_path: str) -> bool:
    """
    Check if a file is an audio file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an audio file
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in AUDIO_EXTENSIONS


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    output_format: str = "mp3",
) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Optional path for the output audio file.
                    If None, creates a temp file.
        output_format: Audio format to export (default: mp3)
        
    Returns:
        Path to the extracted audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Extracting audio from: {os.path.basename(video_path)}")
    
    # Load the video file (pydub/ffmpeg handles the extraction)
    audio = AudioSegment.from_file(video_path)
    
    # Determine output path
    if output_path is None:
        # Create a temp file with the same base name
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"{base_name}_audio.{output_format}")
    
    # Export the audio
    print(f"Saving audio to: {output_path}")
    audio.export(output_path, format=output_format)
    
    duration_min = len(audio) / (1000 * 60)
    print(f"Extracted {duration_min:.1f} minutes of audio")
    
    return output_path


def ensure_audio_file(file_path: str) -> tuple[str, bool]:
    """
    Ensure we have an audio file to work with.
    
    If the input is a video file, extracts the audio.
    If the input is already an audio file, returns it as-is.
    
    Args:
        file_path: Path to audio or video file
        
    Returns:
        Tuple of (audio_file_path, was_extracted)
        was_extracted is True if audio was extracted from video
    """
    if is_audio_file(file_path):
        return file_path, False
    
    if is_video_file(file_path):
        audio_path = extract_audio(file_path)
        return audio_path, True
    
    # Unknown extension - try to process it anyway
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Warning: Unknown file extension '{ext}', attempting to process as audio")
    return file_path, False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <video_file>")
        sys.exit(1)
    
    audio_path = extract_audio(sys.argv[1])
    print(f"\nExtracted audio to: {audio_path}")


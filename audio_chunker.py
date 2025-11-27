"""
Audio chunking module for splitting long audio files into overlapping segments.
"""
import os
import tempfile
from math import ceil
from dataclasses import dataclass
from typing import List, Tuple

from pydub import AudioSegment

from config import (
    MIN_CHUNK_DURATION_MIN,
    MAX_CHUNK_DURATION_MIN,
    MIN_FINAL_CHUNK_MIN,
    OVERLAP_RATIO,
)


@dataclass
class AudioChunk:
    """Represents a chunk of audio with metadata."""
    index: int
    file_path: str
    start_time_ms: int
    end_time_ms: int
    duration_ms: int
    has_overlap_before: bool
    has_overlap_after: bool
    overlap_duration_ms: int


def calculate_chunk_parameters(total_duration_min: float) -> Tuple[float, float]:
    """
    Calculate optimal chunk size and overlap to ensure:
    - Chunks are between MIN_CHUNK_DURATION_MIN and MAX_CHUNK_DURATION_MIN
    - Final chunk is at least MIN_FINAL_CHUNK_MIN minutes
    - Overlap is approximately OVERLAP_RATIO of chunk size
    
    Args:
        total_duration_min: Total duration of the audio in minutes
        
    Returns:
        Tuple of (chunk_duration_min, overlap_duration_min)
    """
    # For very short files, just return the whole thing
    if total_duration_min <= MAX_CHUNK_DURATION_MIN:
        return total_duration_min, 0.0
    
    # Try chunk sizes from MAX down to MIN to find one where last chunk >= MIN_FINAL_CHUNK_MIN
    for chunk_size in range(MAX_CHUNK_DURATION_MIN, MIN_CHUNK_DURATION_MIN - 1, -1):
        overlap = chunk_size * OVERLAP_RATIO
        
        # Calculate effective step (how much we advance per chunk)
        step = chunk_size - overlap
        
        # Calculate number of chunks needed
        # First chunk covers chunk_size, each subsequent chunk adds 'step' more coverage
        num_chunks = 1 + ceil((total_duration_min - chunk_size) / step)
        
        # Calculate where the last chunk would start
        last_chunk_start = (num_chunks - 1) * step
        
        # Calculate last chunk duration
        last_chunk_duration = total_duration_min - last_chunk_start
        
        if last_chunk_duration >= MIN_FINAL_CHUNK_MIN:
            return float(chunk_size), overlap
    
    # Fallback: use minimum chunk size and accept shorter final chunk
    # This handles edge cases with unusual durations
    chunk_size = MIN_CHUNK_DURATION_MIN
    overlap = chunk_size * OVERLAP_RATIO
    return float(chunk_size), overlap


def chunk_audio(audio_path: str, output_dir: str = None) -> List[AudioChunk]:
    """
    Split an audio file into overlapping chunks.
    
    Args:
        audio_path: Path to the input audio file (MP3)
        output_dir: Directory to save chunk files. If None, uses a temp directory.
        
    Returns:
        List of AudioChunk objects with paths to the chunk files
    """
    # Load the audio file
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    
    total_duration_ms = len(audio)
    total_duration_min = total_duration_ms / (1000 * 60)
    
    print(f"Total duration: {total_duration_min:.2f} minutes")
    
    # Calculate chunk parameters
    chunk_duration_min, overlap_duration_min = calculate_chunk_parameters(total_duration_min)
    
    chunk_duration_ms = int(chunk_duration_min * 60 * 1000)
    overlap_duration_ms = int(overlap_duration_min * 60 * 1000)
    step_ms = chunk_duration_ms - overlap_duration_ms
    
    print(f"Chunk size: {chunk_duration_min:.1f} min, Overlap: {overlap_duration_min:.1f} min")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    chunks: List[AudioChunk] = []
    chunk_index = 0
    start_ms = 0
    
    # Minimum duration for a trailing chunk (in ms) - skip tiny remnants
    min_trailing_chunk_ms = 5000  # 5 seconds
    
    while start_ms < total_duration_ms:
        # Calculate remaining audio duration
        remaining_ms = total_duration_ms - start_ms
        
        # Skip if the remaining audio is too short to be worth a separate chunk
        # (This handles cases where audio is e.g. 13.001 minutes and we'd create a 0.1s chunk)
        if chunk_index > 0 and remaining_ms < min_trailing_chunk_ms:
            # Extend the previous chunk to include this tiny remainder
            if chunks:
                prev_chunk = chunks[-1]
                # Re-export the previous chunk with the extended duration
                extended_audio = audio[prev_chunk.start_time_ms:total_duration_ms]
                extended_audio.export(prev_chunk.file_path, format="mp3")
                # Update the chunk metadata
                chunks[-1] = AudioChunk(
                    index=prev_chunk.index,
                    file_path=prev_chunk.file_path,
                    start_time_ms=prev_chunk.start_time_ms,
                    end_time_ms=total_duration_ms,
                    duration_ms=total_duration_ms - prev_chunk.start_time_ms,
                    has_overlap_before=prev_chunk.has_overlap_before,
                    has_overlap_after=False,
                    overlap_duration_ms=prev_chunk.overlap_duration_ms,
                )
                print(f"  Extended chunk {prev_chunk.index} to include {remaining_ms/1000:.1f}s remainder")
            break
        
        # Calculate end position
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        
        # Extract the chunk
        chunk_audio = audio[start_ms:end_ms]
        
        # Generate output filename
        chunk_filename = f"chunk_{chunk_index:03d}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Export the chunk
        chunk_audio.export(chunk_path, format="mp3")
        
        # Create chunk metadata
        chunk = AudioChunk(
            index=chunk_index,
            file_path=chunk_path,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            duration_ms=end_ms - start_ms,
            has_overlap_before=chunk_index > 0,
            has_overlap_after=end_ms < total_duration_ms,
            overlap_duration_ms=overlap_duration_ms if chunk_index > 0 else 0,
        )
        
        chunks.append(chunk)
        print(f"  Created chunk {chunk_index}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s ({(end_ms-start_ms)/1000:.1f}s)")
        
        chunk_index += 1
        start_ms += step_ms
        
        # Break if we've covered all the audio
        if end_ms >= total_duration_ms:
            break
    
    print(f"Created {len(chunks)} chunks in {output_dir}")
    return chunks


def cleanup_chunks(chunks: List[AudioChunk]) -> None:
    """
    Delete temporary chunk files.
    
    Args:
        chunks: List of AudioChunk objects to clean up
    """
    for chunk in chunks:
        if os.path.exists(chunk.file_path):
            os.remove(chunk.file_path)
    
    # Try to remove the directory if it's empty
    if chunks:
        chunk_dir = os.path.dirname(chunks[0].file_path)
        try:
            os.rmdir(chunk_dir)
        except OSError:
            pass  # Directory not empty or other issue


if __name__ == "__main__":
    # Test with a sample file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_chunker.py <audio_file>")
        sys.exit(1)
    
    chunks = chunk_audio(sys.argv[1])
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  {chunk.index}: {chunk.file_path} ({chunk.duration_ms/1000:.1f}s)")


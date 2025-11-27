"""
Main pipeline orchestrator for audio transcription.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

from audio_chunker import chunk_audio, cleanup_chunks, AudioChunk
from audio_extractor import ensure_audio_file, is_video_file
from transcript_stitcher import stitch_transcripts, TranscriptSegment
from providers import TranscriptionProvider, GeminiProvider
from config import TRANSCRIPTION_PROMPT, OUTPUT_DIR


@dataclass
class TranscriptionResult:
    """Result of a transcription pipeline run."""
    input_file: str
    output_file: str
    transcript: str
    num_chunks: int
    total_duration_ms: int
    processing_time_seconds: float
    provider_name: str


class TranscriptionPipeline:
    """
    Orchestrates the audio transcription process:
    1. Chunk the audio file
    2. Transcribe each chunk using the provider
    3. Stitch transcripts together
    4. Save the final output
    """
    
    def __init__(
        self,
        provider: Optional[TranscriptionProvider] = None,
        prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
        keep_chunks: bool = False,
    ):
        """
        Initialize the transcription pipeline.
        
        Args:
            provider: Transcription provider to use. Defaults to GeminiProvider.
            prompt: Custom transcription prompt. Defaults to TRANSCRIPTION_PROMPT.
            output_dir: Directory for output files. Defaults to OUTPUT_DIR.
            keep_chunks: If True, don't delete temporary chunk files.
        """
        self.provider = provider or GeminiProvider()
        self.prompt = prompt or TRANSCRIPTION_PROMPT
        self.output_dir = output_dir or OUTPUT_DIR
        self.keep_chunks = keep_chunks
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def transcribe_file(
        self,
        input_path: str,
        output_filename: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio or video file end-to-end.
        
        Args:
            input_path: Path to the input audio or video file
            output_filename: Optional custom output filename
            
        Returns:
            TranscriptionResult with the transcript and metadata
        """
        start_time = time.time()
        original_input = input_path
        extracted_audio_path = None
        
        print(f"\n{'='*60}")
        print(f"Transcription Pipeline")
        print(f"{'='*60}")
        print(f"Input: {input_path}")
        print(f"Provider: {self.provider.name}")
        print(f"{'='*60}\n")
        
        # Step 0: Extract audio from video if needed
        if is_video_file(input_path):
            print("Step 0: Extracting audio from video...")
            audio_path, was_extracted = ensure_audio_file(input_path)
            if was_extracted:
                extracted_audio_path = audio_path
            print()
        else:
            audio_path = input_path
        
        # Step 1: Chunk the audio
        print("Step 1: Chunking audio...")
        chunks = chunk_audio(audio_path)
        
        if not chunks:
            raise ValueError("No audio chunks were created")
        
        total_duration_ms = chunks[-1].end_time_ms
        
        # Step 2: Transcribe each chunk
        print(f"\nStep 2: Transcribing {len(chunks)} chunks...")
        segments = self._transcribe_chunks(chunks)
        
        # Step 3: Stitch transcripts together
        print(f"\nStep 3: Stitching transcripts...")
        final_transcript = stitch_transcripts(segments)
        
        # Step 4: Save output
        print(f"\nStep 4: Saving output...")
        output_path = self._save_transcript(original_input, final_transcript, output_filename)
        
        # Cleanup
        if not self.keep_chunks:
            print(f"\nCleaning up temporary files...")
            cleanup_chunks(chunks)
        
        # Clean up extracted audio if we created one
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            os.remove(extracted_audio_path)
            print(f"Removed extracted audio: {extracted_audio_path}")
        
        processing_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Transcription Complete!")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Chunks processed: {len(chunks)}")
        print(f"Total duration: {total_duration_ms/1000/60:.1f} minutes")
        print(f"Processing time: {processing_time:.1f} seconds")
        print(f"{'='*60}\n")
        
        return TranscriptionResult(
            input_file=original_input,
            output_file=output_path,
            transcript=final_transcript,
            num_chunks=len(chunks),
            total_duration_ms=total_duration_ms,
            processing_time_seconds=processing_time,
            provider_name=self.provider.name,
        )
    
    def _transcribe_chunks(self, chunks: List[AudioChunk]) -> List[TranscriptSegment]:
        """
        Transcribe all audio chunks.
        
        Args:
            chunks: List of AudioChunk objects
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n  Transcribing chunk {i + 1}/{len(chunks)}...")
            print(f"    File: {os.path.basename(chunk.file_path)}")
            print(f"    Duration: {chunk.duration_ms/1000:.1f}s")
            
            try:
                transcript = self.provider.transcribe(chunk.file_path, self.prompt)
                
                segment = TranscriptSegment(
                    chunk_index=chunk.index,
                    text=transcript,
                    overlap_duration_ms=chunk.overlap_duration_ms,
                    has_overlap_before=chunk.has_overlap_before,
                    has_overlap_after=chunk.has_overlap_after,
                )
                segments.append(segment)
                
                print(f"    Transcribed {len(transcript)} characters")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                # Add empty segment to maintain order
                segments.append(TranscriptSegment(
                    chunk_index=chunk.index,
                    text=f"[Transcription failed for chunk {chunk.index}: {e}]",
                    overlap_duration_ms=chunk.overlap_duration_ms,
                    has_overlap_before=chunk.has_overlap_before,
                    has_overlap_after=chunk.has_overlap_after,
                ))
        
        return segments
    
    def _save_transcript(
        self,
        audio_path: str,
        transcript: str,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Save the transcript to a file.
        
        Args:
            audio_path: Original audio file path (for naming)
            transcript: The transcript text
            output_filename: Optional custom filename
            
        Returns:
            Path to the saved transcript file
        """
        if output_filename:
            filename = output_filename
        else:
            # Generate filename from audio file
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_transcript_{timestamp}.txt"
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        print(f"    Saved to: {output_path}")
        return output_path


def transcribe(
    audio_path: str,
    provider: Optional[TranscriptionProvider] = None,
    prompt: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    keep_chunks: bool = False,
) -> TranscriptionResult:
    """
    Convenience function to transcribe an audio file.
    
    Args:
        audio_path: Path to the audio file
        provider: Optional transcription provider
        prompt: Optional custom prompt
        output_dir: Optional output directory
        output_filename: Optional output filename
        keep_chunks: Whether to keep temporary chunk files
        
    Returns:
        TranscriptionResult with the transcript and metadata
    """
    pipeline = TranscriptionPipeline(
        provider=provider,
        prompt=prompt,
        output_dir=output_dir,
        keep_chunks=keep_chunks,
    )
    
    return pipeline.transcribe_file(audio_path, output_filename)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)
    
    result = transcribe(sys.argv[1])
    print(f"\nTranscript saved to: {result.output_file}")


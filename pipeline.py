"""
Main pipeline orchestrator for audio transcription.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

from audio_chunker import chunk_audio, cleanup_chunks, AudioChunk
from audio_extractor import ensure_audio_file, is_video_file
from transcript_stitcher import stitch_transcripts, TranscriptSegment
from speaker_reconciler import (
    extract_transcript_and_metadata,
    reconcile_speakers,
    apply_speaker_mapping,
    SpeakerMetadata,
)
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
        raw_transcripts, all_speaker_metadata = self._transcribe_chunks_with_metadata(chunks)
        
        # Step 3: Reconcile speakers across chunks
        print(f"\nStep 3: Reconciling speakers across chunks...")
        reconciled_transcripts = self._reconcile_and_apply_speakers(
            raw_transcripts, all_speaker_metadata
        )
        
        # Step 4: Build segments and stitch transcripts together
        print(f"\nStep 4: Stitching transcripts...")
        segments = [
            TranscriptSegment(
                chunk_index=i,
                text=transcript,
                overlap_duration_ms=chunks[i].overlap_duration_ms,
                has_overlap_before=chunks[i].has_overlap_before,
                has_overlap_after=chunks[i].has_overlap_after,
            )
            for i, transcript in enumerate(reconciled_transcripts)
        ]
        final_transcript = stitch_transcripts(segments)
        
        # Step 5: Save output
        print(f"\nStep 5: Saving output...")
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
    
    def _transcribe_chunks_with_metadata(
        self, chunks: List[AudioChunk]
    ) -> tuple[List[str], Dict[int, List[SpeakerMetadata]]]:
        """
        Transcribe all audio chunks and extract speaker metadata.
        
        Args:
            chunks: List of AudioChunk objects
            
        Returns:
            Tuple of (list of transcript texts, dict of chunk_index to speaker metadata)
        """
        transcripts = []
        all_metadata: Dict[int, List[SpeakerMetadata]] = {}
        
        for i, chunk in enumerate(chunks):
            print(f"\n  Transcribing chunk {i + 1}/{len(chunks)}...")
            print(f"    File: {os.path.basename(chunk.file_path)}")
            print(f"    Duration: {chunk.duration_ms/1000:.1f}s")
            
            try:
                raw_response = self.provider.transcribe(chunk.file_path, self.prompt)
                
                # Extract transcript and speaker metadata
                transcript, speaker_metadata = extract_transcript_and_metadata(
                    raw_response, chunk.index
                )
                
                transcripts.append(transcript)
                if speaker_metadata:
                    all_metadata[chunk.index] = speaker_metadata
                    print(f"    Found {len(speaker_metadata)} speakers")
                
                print(f"    Transcribed {len(transcript)} characters")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                transcripts.append(f"[Transcription failed for chunk {chunk.index}: {e}]")
        
        return transcripts, all_metadata
    
    def _reconcile_and_apply_speakers(
        self,
        transcripts: List[str],
        all_metadata: Dict[int, List[SpeakerMetadata]]
    ) -> List[str]:
        """
        Reconcile speakers across chunks and apply consistent labels.
        
        Args:
            transcripts: List of transcript texts
            all_metadata: Dict mapping chunk index to speaker metadata
            
        Returns:
            List of transcripts with consistent speaker labels
        """
        if not all_metadata:
            print("  No speaker metadata found, skipping reconciliation")
            return transcripts
        
        try:
            # Get the speaker mapping from LLM
            replacement_map = reconcile_speakers(all_metadata, self.provider)
            
            # Apply mapping to each transcript
            reconciled = []
            for i, transcript in enumerate(transcripts):
                if i in replacement_map:
                    mapped_transcript = apply_speaker_mapping(
                        transcript, replacement_map[i]
                    )
                    reconciled.append(mapped_transcript)
                else:
                    reconciled.append(transcript)
            
            return reconciled
            
        except Exception as e:
            print(f"  Warning: Speaker reconciliation failed: {e}")
            print("  Falling back to original speaker labels")
            return transcripts
    
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


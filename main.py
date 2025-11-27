"""
CLI entry point for the audio transcription pipeline.
"""
import argparse
import sys
import os

from pipeline import TranscriptionPipeline, TranscriptionResult
from providers import GeminiProvider
from config import TRANSCRIPTION_PROMPT, OUTPUT_DIR


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Transcribe long audio or video files using LLM APIs with intelligent chunking and stitching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio.mp3
  python main.py video.mp4                  # Extracts audio automatically
  python main.py audio.mp3 --output-dir transcripts/
  python main.py audio.mp3 --keep-chunks --output transcript.txt
  python main.py audio.mp3 --provider gemini

Environment Variables:
  GEMINI_API_KEY    Google Gemini API key (required for Gemini provider)
  OUTPUT_DIR        Default output directory (optional)
        """,
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the audio or video file to transcribe (MP3, WAV, M4A, MP4, etc.)",
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output_filename",
        help="Output filename for the transcript (default: auto-generated)",
    )
    
    parser.add_argument(
        "-d", "--output-dir",
        dest="output_dir",
        default=OUTPUT_DIR,
        help=f"Directory for output files (default: {OUTPUT_DIR})",
    )
    
    parser.add_argument(
        "-p", "--provider",
        choices=["gemini"],  # Add more providers here as they're implemented
        default="gemini",
        help="Transcription provider to use (default: gemini)",
    )
    
    parser.add_argument(
        "--prompt",
        help="Custom transcription prompt (default: built-in speaker diarization prompt)",
    )
    
    parser.add_argument(
        "--prompt-file",
        help="Path to a file containing the transcription prompt",
    )
    
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep temporary audio chunk files after processing",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser


def get_provider(provider_name: str):
    """
    Get the transcription provider by name.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        TranscriptionProvider instance
    """
    providers = {
        "gemini": GeminiProvider,
        # Add more providers here as they're implemented
        # "anthropic": AnthropicProvider,
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return providers[provider_name]()


def get_prompt(args) -> str:
    """
    Get the transcription prompt from args or defaults.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        The prompt string
    """
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    if args.prompt:
        return args.prompt
    
    return TRANSCRIPTION_PROMPT


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Get provider
        provider = get_provider(args.provider)
        
        # Get prompt
        prompt = get_prompt(args)
        
        # Create pipeline
        pipeline = TranscriptionPipeline(
            provider=provider,
            prompt=prompt,
            output_dir=args.output_dir,
            keep_chunks=args.keep_chunks,
        )
        
        # Run transcription
        result = pipeline.transcribe_file(
            args.input_file,
            output_filename=args.output_filename,
        )
        
        # Print summary
        print(f"\nTranscription complete!")
        print(f"  Output file: {result.output_file}")
        print(f"  Audio duration: {result.total_duration_ms/1000/60:.1f} minutes")
        print(f"  Chunks processed: {result.num_chunks}")
        print(f"  Processing time: {result.processing_time_seconds:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\nTranscription cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


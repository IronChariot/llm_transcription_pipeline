(fieri iussit, created using Cursor + Opus 4.5/Gemini 3.0 Pro)

# Transcription Pipeline

A robust audio transcription and diarization pipeline designed for long audio and video files. It uses Large Language Models (LLMs) to transcribe content with high accuracy, identifying speakers and maintaining context across long recordings.

## Features

*   **Long-form Transcription**: Handles files of any length by intelligently chunking audio.
*   **Speaker Diarization**: Identifies and separates different speakers (e.g., **Speaker 1**, **Speaker 2**).
*   **Smart Speaker Reconciliation**: Analyzes voice descriptions, roles, and verbal patterns to consistently label speakers across different audio chunks (e.g., ensuring "Speaker 1" in chunk 1 is identified as the same person as "Speaker 3" in chunk 2).
*   **Video Support**: Automatically extracts audio from video files (MP4, etc.).
*   **Detailed Logging**: Keeps comprehensive logs of the process for debugging and verification.
*   **LLM Powered**: Currently supports Google Gemini models for state-of-the-art transcription accuracy.

## Prerequisites

1.  **Python 3.8+**
2.  **FFmpeg**: Required for audio processing.
    *   **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your system PATH.
    *   **Mac**: `brew install ffmpeg`
    *   **Linux**: `sudo apt install ffmpeg`

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd transcription_pipeline
    ```

2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your environment variables. Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_google_gemini_api_key
    ```
    (You can get an API key from [Google AI Studio](https://aistudio.google.com/))

## Usage

Run the pipeline via the command line interface:

```bash
# Basic usage
python main.py path/to/audio.mp3

# Video file (audio is extracted automatically)
python main.py path/to/video.mp4

# Specify output directory
python main.py meeting.mp3 --output-dir my_transcripts/

# Keep temporary audio chunks (useful for debugging)
python main.py interview.wav --keep-chunks

# Show help
python main.py --help
```

### CLI Options

*   `input_file`: Path to the audio/video file (Required).
*   `-o`, `--output`: Custom output filename.
*   `-d`, `--output-dir`: Directory for output files (Default: `output/`).
*   `-p`, `--provider`: AI provider to use (Default: `gemini`).
*   `--keep-chunks`: Don't delete temporary audio chunks after processing.
*   `-v`, `--verbose`: Enable verbose output.

## How It Works

The pipeline follows these steps:

1.  **Extraction**: If a video file is provided, the audio track is extracted using FFmpeg.
2.  **Chunking**: The audio is split into overlapping chunks (default 10-20 mins) to fit within LLM context windows.
3.  **Transcription**: Each chunk is sent to the LLM. The model provides:
    *   A verbatim transcript with speaker labels.
    *   Metadata about each speaker (voice description, estimated role, verbal patterns).
4.  **Reconciliation**: The system gathers speaker metadata from all chunks and uses an LLM to figure out who is who across the entire file, mapping local "Speaker X" labels to consistent global "Person Y" labels.
5.  **Stitching**: Transcripts are stitched together. Overlapping regions are carefully aligned to ensure no text is lost or duplicated.
6.  **Output**: The final reconciled transcript is saved to the output directory.

## Output & Logs

### Final Transcript
Saved in `output/` (or your specified directory) as a text file:
*   `filename_transcript_TIMESTAMP.txt`

### Process Logs
Detailed logs are saved in `process_logs/<filename>_TIMESTAMP/`. This is useful for understanding how the AI made decisions.

*   **`chunks/` directory**: Contains data for each processed audio chunk.
    *   `chunk_XXX_transcript.txt`: The raw transcript for that specific chunk (before speaker reconciliation).
    *   `chunk_XXX_speakers.json`: The extracted metadata for speakers found in this chunk (voice, role, patterns).
    *   `chunk_XXX_raw_response.txt`: The full raw text response from the LLM.

*   **Reconciliation Files**:
    *   `reconciliation_input.json`: The compiled list of all speakers from all chunks sent to the LLM for matching.
    *   `reconciliation_mappings.json`: The decision logic—how the system mapped each chunk's "Speaker X" to the final "Person Y".
    *   `reconciliation_response.txt`: The raw explanation from the LLM about why it matched speakers the way it did.
    *   `speaker_debug_summary.json`: A human-readable summary of the final merged speaker profiles. Particularly useful for doing a final find+replace on the transcript file to swap in people's real names.

## Configuration

You can adjust core settings in `config.py`:

*   `MIN_CHUNK_DURATION_MIN`: Minimum length of audio chunks.
*   `MAX_CHUNK_DURATION_MIN`: Maximum length of audio chunks.
*   `OVERLAP_RATIO`: How much chunks should overlap (to prevent cutting words at boundaries).
*   `TRANSCRIPTION_PROMPT`: The system prompt used to instruct the LLM on formatting and style.


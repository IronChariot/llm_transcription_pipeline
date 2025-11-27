"""
Transcript stitching module for merging overlapping transcript segments.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with metadata."""
    chunk_index: int
    text: str
    overlap_duration_ms: int = 0
    has_overlap_before: bool = False
    has_overlap_after: bool = False


def normalize_speaker_label(label: str) -> str:
    """
    Normalize speaker labels to a consistent format.
    
    Args:
        label: Original speaker label (e.g., "Speaker 1:", "SPEAKER A:", "Person One:")
        
    Returns:
        Normalized label in format "Speaker N:"
    """
    # Remove common prefixes and clean up
    label = label.strip().rstrip(':')
    label_lower = label.lower()
    
    # Extract number or letter from common patterns
    patterns = [
        r'speaker\s*(\d+)',
        r'speaker\s*([a-z])',
        r'person\s*(\d+)',
        r'person\s*([a-z])',
        r'voice\s*(\d+)',
        r'voice\s*([a-z])',
        r'^([a-z])$',
        r'^(\d+)$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, label_lower)
        if match:
            identifier = match.group(1)
            # Convert letter to number if needed
            if identifier.isalpha():
                identifier = str(ord(identifier.upper()) - ord('A') + 1)
            return f"Speaker {identifier}"
    
    # If no pattern matched, return cleaned original
    return label


def extract_lines_with_speakers(text: str) -> List[Tuple[str, str]]:
    """
    Extract lines from transcript, separating speaker labels from content.
    
    Args:
        text: Raw transcript text
        
    Returns:
        List of (speaker_label, content) tuples
    """
    lines = []
    
    # Pattern to match speaker labels
    # Matches: "Speaker 1:", "SPEAKER A:", "Person One:", "John:", etc.
    speaker_pattern = re.compile(r'^([A-Za-z][A-Za-z0-9\s]*?):\s*(.*)$')
    
    current_speaker = None
    current_content = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = speaker_pattern.match(line)
        if match:
            # Save previous speaker's content
            if current_speaker and current_content:
                lines.append((current_speaker, ' '.join(current_content)))
            
            current_speaker = match.group(1)
            content = match.group(2).strip()
            current_content = [content] if content else []
        elif current_speaker:
            # Continuation of current speaker
            current_content.append(line)
    
    # Don't forget the last speaker
    if current_speaker and current_content:
        lines.append((current_speaker, ' '.join(current_content)))
    
    return lines


def find_overlap_alignment(
    text1: str,
    text2: str,
    overlap_ratio: float = 0.15,
    min_match_score: float = 60.0
) -> Tuple[int, int, float]:
    """
    Find the best alignment point between the end of text1 and start of text2.
    
    Uses fuzzy matching to handle slight transcription differences.
    
    Args:
        text1: First transcript (look at the end)
        text2: Second transcript (look at the beginning)
        overlap_ratio: Approximate ratio of text that overlaps
        min_match_score: Minimum fuzzy match score to consider a match
        
    Returns:
        Tuple of (split_point_in_text1, split_point_in_text2, match_score)
        Returns (-1, -1, 0) if no good match found
    """
    # Get words from both texts
    words1 = text1.split()
    words2 = text2.split()
    
    if not words1 or not words2:
        return -1, -1, 0.0
    
    # Estimate overlap region in words
    estimated_overlap_words = int(len(words2) * overlap_ratio)
    search_window = max(estimated_overlap_words * 2, 50)  # Search window
    
    # Look at the end of text1
    end_words1 = words1[-search_window:] if len(words1) > search_window else words1
    # Look at the start of text2
    start_words2 = words2[:search_window] if len(words2) > search_window else words2
    
    best_score = 0.0
    best_i = -1
    best_j = -1
    
    # Sliding window comparison
    # Try different phrase lengths for matching
    for phrase_len in [10, 15, 20, 8, 5]:
        if phrase_len > len(end_words1) or phrase_len > len(start_words2):
            continue
            
        for i in range(len(end_words1) - phrase_len + 1):
            phrase1 = ' '.join(end_words1[i:i + phrase_len])
            
            for j in range(len(start_words2) - phrase_len + 1):
                phrase2 = ' '.join(start_words2[j:j + phrase_len])
                
                # Calculate similarity
                score = fuzz.ratio(phrase1.lower(), phrase2.lower())
                
                if score > best_score and score >= min_match_score:
                    best_score = score
                    # Calculate actual positions in full texts
                    words1_offset = len(words1) - len(end_words1)
                    best_i = words1_offset + i
                    best_j = j
    
    if best_score >= min_match_score:
        # Convert word positions back to character positions
        char_pos1 = len(' '.join(words1[:best_i]))
        char_pos2 = len(' '.join(words2[:best_j]))
        return char_pos1, char_pos2, best_score
    
    return -1, -1, 0.0


def build_speaker_mapping(
    lines1: List[Tuple[str, str]],
    lines2: List[Tuple[str, str]],
    overlap_content1: str,
    overlap_content2: str
) -> Dict[str, str]:
    """
    Build a mapping from speakers in text2 to speakers in text1.
    
    Uses the overlapping content to identify which speakers correspond.
    
    Args:
        lines1: Speaker lines from first transcript
        lines2: Speaker lines from second transcript
        overlap_content1: The overlapping portion from transcript 1
        overlap_content2: The overlapping portion from transcript 2
        
    Returns:
        Dict mapping speaker labels from text2 to text1
    """
    mapping = {}
    
    # Get unique speakers from each
    speakers1 = set(normalize_speaker_label(s) for s, _ in lines1)
    speakers2 = set(normalize_speaker_label(s) for s, _ in lines2)
    
    # For each speaker in text2's overlap, find best match in text1's overlap
    # Based on content similarity
    
    # Extract what each speaker says in the overlap regions
    speaker_content1 = {}
    speaker_content2 = {}
    
    for speaker, content in lines1:
        norm_speaker = normalize_speaker_label(speaker)
        # Check if this content appears in the overlap region
        if content.lower() in overlap_content1.lower():
            speaker_content1[norm_speaker] = speaker_content1.get(norm_speaker, '') + ' ' + content
    
    for speaker, content in lines2:
        norm_speaker = normalize_speaker_label(speaker)
        if content.lower() in overlap_content2.lower():
            speaker_content2[norm_speaker] = speaker_content2.get(norm_speaker, '') + ' ' + content
    
    # Match speakers based on content similarity
    for s2, content2 in speaker_content2.items():
        best_match = None
        best_score = 0
        
        for s1, content1 in speaker_content1.items():
            score = fuzz.ratio(content1.lower(), content2.lower())
            if score > best_score:
                best_score = score
                best_match = s1
        
        if best_match and best_score > 50:
            mapping[s2] = best_match
    
    # For speakers not in overlap, map by order/number
    unmapped2 = speakers2 - set(mapping.keys())
    unused1 = speakers1 - set(mapping.values())
    
    # Sort for consistent ordering
    unmapped2 = sorted(unmapped2)
    unused1 = sorted(unused1)
    
    for s2, s1 in zip(unmapped2, unused1):
        mapping[s2] = s1
    
    return mapping


def apply_speaker_mapping(text: str, mapping: Dict[str, str]) -> str:
    """
    Apply speaker mapping to normalize labels in a transcript.
    
    Args:
        text: Transcript text
        mapping: Dict mapping old speaker labels to new ones
        
    Returns:
        Text with updated speaker labels
    """
    result = text
    
    for old_label, new_label in mapping.items():
        # Match the speaker label at the start of lines
        pattern = re.compile(rf'^{re.escape(old_label)}:', re.MULTILINE)
        result = pattern.sub(f'{new_label}:', result)
    
    return result


def stitch_transcripts(segments: List[TranscriptSegment]) -> str:
    """
    Stitch together multiple overlapping transcript segments.
    
    Args:
        segments: List of TranscriptSegment objects in order
        
    Returns:
        Combined transcript with consistent speaker labels
    """
    if not segments:
        return ""
    
    if len(segments) == 1:
        return segments[0].text
    
    # Start with the first segment
    result = segments[0].text
    
    # Process each subsequent segment
    for i in range(1, len(segments)):
        prev_segment = segments[i - 1]
        curr_segment = segments[i]
        
        print(f"  Stitching chunk {i-1} with chunk {i}...")
        
        # Find alignment in overlap region
        split1, split2, score = find_overlap_alignment(
            result,
            curr_segment.text,
            overlap_ratio=0.10
        )
        
        if split1 >= 0 and split2 >= 0:
            print(f"    Found alignment with score {score:.1f}")
            
            # Extract overlap regions for speaker mapping
            overlap1 = result[split1:]
            overlap2 = curr_segment.text[:split2 + len(result) - split1]
            
            # Build speaker mapping
            lines1 = extract_lines_with_speakers(result)
            lines2 = extract_lines_with_speakers(curr_segment.text)
            
            speaker_mapping = build_speaker_mapping(
                lines1, lines2,
                overlap1, overlap2
            )
            
            if speaker_mapping:
                print(f"    Speaker mapping: {speaker_mapping}")
            
            # Apply mapping to current segment
            mapped_text = apply_speaker_mapping(curr_segment.text, speaker_mapping)
            
            # Stitch: take text1 up to split point, then text2 from split point
            result = result[:split1].rstrip() + '\n\n' + mapped_text[split2:].lstrip()
        else:
            print(f"    No good alignment found, appending with separator")
            # No good alignment found - just append with a separator
            result = result.rstrip() + '\n\n---\n\n' + curr_segment.text.lstrip()
    
    # Clean up extra whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


if __name__ == "__main__":
    # Test with sample transcripts
    sample1 = """Speaker 1: Hello everyone, welcome to the meeting.
Speaker 2: Thanks for having us here today.
Speaker 1: Let's get started with the first topic.
Speaker 2: Sure, I'd like to discuss the project timeline.
Speaker 1: The project timeline looks good so far."""

    sample2 = """Speaker A: The project timeline looks good so far.
Speaker B: I agree, we're on track to meet our goals.
Speaker A: Great, let's move on to the next item.
Speaker B: The budget review is next on the agenda."""

    segments = [
        TranscriptSegment(0, sample1, has_overlap_after=True),
        TranscriptSegment(1, sample2, has_overlap_before=True),
    ]
    
    result = stitch_transcripts(segments)
    print("--- Stitched Result ---")
    print(result)


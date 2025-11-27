"""
Speaker reconciliation module for consistent speaker labels across chunks.
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from config import RECONCILIATION_PROMPT


@dataclass
class SpeakerMetadata:
    """Metadata about a speaker from a single chunk."""
    label: str
    voice_description: str
    role_estimation: str
    chunk_index: int


@dataclass
class PersonMapping:
    """Mapping of a person to their labels in each chunk."""
    master_label: str
    voice_description: str = ""
    role_estimation: str = ""
    chunk_labels: List[Dict[str, Any]] = field(default_factory=list)


def extract_transcript_and_metadata(
    raw_response: str,
    chunk_index: int
) -> Tuple[str, List[SpeakerMetadata]]:
    """
    Parse LLM response to separate transcript text from JSON metadata block.
    
    Args:
        raw_response: Raw response from the LLM containing transcript and JSON
        chunk_index: Index of the chunk this response belongs to
        
    Returns:
        Tuple of (transcript_text, list of SpeakerMetadata)
    """
    # Look for JSON code block at the end
    json_pattern = r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```'
    
    matches = list(re.finditer(json_pattern, raw_response))
    
    if not matches:
        # No JSON block found - return full response as transcript
        print(f"  Warning: No speaker metadata JSON found in chunk {chunk_index}")
        return raw_response.strip(), []
    
    # Use the last JSON block (should be the speaker metadata)
    last_match = matches[-1]
    json_str = last_match.group(1)
    
    # Extract transcript (everything before the last JSON block)
    transcript = raw_response[:last_match.start()].strip()
    
    # Parse JSON
    try:
        metadata_list = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse speaker metadata JSON in chunk {chunk_index}: {e}")
        return transcript, []
    
    # Convert to SpeakerMetadata objects
    speakers = []
    for item in metadata_list:
        if isinstance(item, dict) and "label" in item:
            speakers.append(SpeakerMetadata(
                label=item.get("label", "Unknown"),
                voice_description=item.get("voice_description", ""),
                role_estimation=item.get("role_estimation", ""),
                chunk_index=chunk_index,
            ))
    
    return transcript, speakers


def build_reconciliation_input(
    all_chunk_metadata: Dict[int, List[SpeakerMetadata]]
) -> str:
    """
    Build the JSON input for the reconciliation LLM call.
    
    Args:
        all_chunk_metadata: Dict mapping chunk index to list of SpeakerMetadata
        
    Returns:
        JSON string formatted for the reconciliation prompt
    """
    chunks_data = {}
    
    for chunk_idx, speakers in sorted(all_chunk_metadata.items()):
        chunk_key = f"chunk_{chunk_idx + 1}"  # 1-indexed for readability
        chunks_data[chunk_key] = [
            {
                "label": s.label,
                "voice_description": s.voice_description,
                "role_estimation": s.role_estimation,
            }
            for s in speakers
        ]
    
    return json.dumps(chunks_data, indent=2)


def parse_reconciliation_response(
    response: str
) -> List[PersonMapping]:
    """
    Parse the LLM's reconciliation response to extract person mappings.
    
    Args:
        response: Raw response from reconciliation LLM call
        
    Returns:
        List of PersonMapping objects
    """
    # Extract JSON from response
    json_pattern = r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```'
    match = re.search(json_pattern, response)
    
    if not match:
        # Try to find raw JSON array
        json_pattern_raw = r'\[\s*\{[\s\S]*\}\s*\]'
        match = re.search(json_pattern_raw, response)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("No valid JSON mapping found in reconciliation response")
    else:
        json_str = match.group(1)
    
    try:
        mappings_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse reconciliation JSON: {e}")
    
    mappings = []
    for item in mappings_data:
        if isinstance(item, dict) and "master_label" in item:
            mapping = PersonMapping(
                master_label=item["master_label"],
                voice_description=item.get("voice_description", ""),
                role_estimation=item.get("role_estimation", ""),
                chunk_labels=item.get("chunk_labels", []),
            )
            mappings.append(mapping)
    
    return mappings


def build_label_replacement_map(
    mappings: List[PersonMapping]
) -> Dict[int, Dict[str, str]]:
    """
    Build a mapping from (chunk_index, old_label) to new_label.
    
    Args:
        mappings: List of PersonMapping objects from reconciliation
        
    Returns:
        Dict[chunk_index, Dict[old_label, new_label]]
    """
    replacement_map: Dict[int, Dict[str, str]] = {}
    
    for person in mappings:
        for chunk_label in person.chunk_labels:
            chunk_idx = chunk_label["chunk"] - 1  # Convert to 0-indexed
            old_label = chunk_label["label"]
            new_label = person.master_label
            
            if chunk_idx not in replacement_map:
                replacement_map[chunk_idx] = {}
            
            replacement_map[chunk_idx][old_label] = new_label
    
    return replacement_map


def apply_speaker_mapping(
    transcript: str,
    mapping: Dict[str, str]
) -> str:
    """
    Replace speaker labels in a transcript according to the mapping.
    
    Args:
        transcript: The transcript text with original speaker labels
        mapping: Dict mapping old labels to new labels (e.g., {"Speaker 1": "Person 2"})
        
    Returns:
        Transcript with updated speaker labels
    """
    result = transcript
    
    # Sort by label length descending to avoid partial replacements
    # (e.g., "Speaker 10" before "Speaker 1")
    sorted_labels = sorted(mapping.keys(), key=len, reverse=True)
    
    for old_label in sorted_labels:
        new_label = mapping[old_label]
        
        # Replace bold format: **Speaker N:** -> **Person M:**
        old_bold = f"**{old_label}:**"
        new_bold = f"**{new_label}:**"
        result = result.replace(old_bold, new_bold)
        
        # Also handle non-bold format just in case: Speaker N: -> **Person M:**
        old_plain = f"{old_label}:"
        # Only replace if not already bold (negative lookbehind for **)
        result = re.sub(
            rf'(?<!\*\*){re.escape(old_plain)}',
            new_bold,
            result
        )
    
    return result


def validate_mapping_coverage(
    mappings: List[PersonMapping],
    all_chunk_metadata: Dict[int, List[SpeakerMetadata]]
) -> Tuple[bool, List[str]]:
    """
    Validate that the mapping covers all speakers in all chunks.
    
    Args:
        mappings: List of PersonMapping from reconciliation
        all_chunk_metadata: Original speaker metadata from all chunks
        
    Returns:
        Tuple of (is_valid, list of missing speaker descriptions)
    """
    # Build set of all (chunk_idx, label) pairs from mappings
    covered = set()
    for person in mappings:
        for chunk_label in person.chunk_labels:
            chunk_idx = chunk_label["chunk"] - 1  # 0-indexed
            label = chunk_label["label"]
            covered.add((chunk_idx, label))
    
    # Check all original speakers are covered
    missing = []
    for chunk_idx, speakers in all_chunk_metadata.items():
        for speaker in speakers:
            if (chunk_idx, speaker.label) not in covered:
                missing.append(f"Chunk {chunk_idx + 1}, {speaker.label}")
    
    return len(missing) == 0, missing


def reconcile_speakers(
    all_chunk_metadata: Dict[int, List[SpeakerMetadata]],
    provider: Any,  # TranscriptionProvider
) -> Dict[int, Dict[str, str]]:
    """
    Reconcile speakers across all chunks using an LLM call.
    
    Args:
        all_chunk_metadata: Dict mapping chunk index to list of SpeakerMetadata
        provider: The transcription provider to use for the LLM call
        
    Returns:
        Dict[chunk_index, Dict[old_label, new_label]] for label replacement
    """
    if not all_chunk_metadata:
        return {}
    
    # Check if there's only one chunk - no reconciliation needed
    if len(all_chunk_metadata) == 1:
        chunk_idx = list(all_chunk_metadata.keys())[0]
        speakers = all_chunk_metadata[chunk_idx]
        # Just map Speaker N -> Person N
        return {
            chunk_idx: {
                s.label: s.label.replace("Speaker", "Person")
                for s in speakers
            }
        }
    
    print("\n  Building speaker reconciliation request...")
    
    # Build the input JSON
    chunk_metadata_json = build_reconciliation_input(all_chunk_metadata)
    
    # Format the prompt
    prompt = RECONCILIATION_PROMPT.format(chunk_metadata=chunk_metadata_json)
    
    print("  Calling LLM for speaker reconciliation...")
    
    # We need to make a text-only call, not an audio call
    # Use the provider's client directly for a text generation
    from google import genai
    from google.genai import types
    
    response = provider.client.models.generate_content(
        model=provider.model,
        contents=[types.Content(parts=[types.Part.from_text(text=prompt)])]
    )
    
    if not response.text:
        raise ValueError("Empty response from reconciliation LLM call")
    
    print("  Parsing reconciliation response...")
    
    # Parse the response
    mappings = parse_reconciliation_response(response.text)
    
    # Validate coverage
    is_valid, missing = validate_mapping_coverage(mappings, all_chunk_metadata)
    
    if not is_valid:
        print(f"  Warning: Reconciliation mapping incomplete. Missing: {missing}")
        # Add missing speakers as "Unknown Person"
        for missing_desc in missing:
            # Parse "Chunk N, Speaker M" format
            match = re.match(r"Chunk (\d+), (.+)", missing_desc)
            if match:
                chunk_idx = int(match.group(1)) - 1
                label = match.group(2)
                # Add to a catch-all mapping
                mappings.append(PersonMapping(
                    master_label="Unknown Person",
                    chunk_labels=[{"chunk": chunk_idx + 1, "label": label}]
                ))
    
    # Build replacement map
    replacement_map = build_label_replacement_map(mappings)
    
    print(f"  Identified {len(mappings)} unique speakers across {len(all_chunk_metadata)} chunks")
    
    return replacement_map


if __name__ == "__main__":
    # Test extraction
    sample_response = """**Speaker 1:** Hello everyone, welcome to the meeting.

**Speaker 2:** Thanks for having us.

**Speaker 1:** Let's get started.

```json
[
  {
    "label": "Speaker 1",
    "voice_description": "male, british accent, baritone",
    "role_estimation": "meeting host, project manager"
  },
  {
    "label": "Speaker 2",
    "voice_description": "female, american accent, soprano",
    "role_estimation": "team member, developer"
  }
]
```"""

    transcript, metadata = extract_transcript_and_metadata(sample_response, 0)
    print("Transcript:")
    print(transcript)
    print("\nMetadata:")
    for m in metadata:
        print(f"  {m.label}: {m.voice_description} - {m.role_estimation}")


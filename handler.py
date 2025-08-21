import os
import subprocess
import runpod
import requests
import tempfile
import json
import torch
from datetime import timedelta
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# Global variables to cache the model (loaded once per container)
alignment_model = None
alignment_tokenizer = None
device = None

def initialize_model():
    """Initialize the alignment model once per container"""
    global alignment_model, alignment_tokenizer, device
    
    if alignment_model is None:
        print("Initializing alignment model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        alignment_model, alignment_tokenizer = load_alignment_model(
            device,
            model_path="facebook/mms-1b-all",
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        print(f"Model initialized on {device}")

def download_file(url, suffix='.mp3'):
    """Download a file from a URL to a temporary file and return its path."""
    try:
        response = requests.get(url, timeout=300)
        if response.status_code != 200:
            raise Exception(f"Failed to download file from URL: {response.status_code}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="./tmp")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise Exception(f"Error downloading file: {str(e)}")

def create_segments_from_lines(timed_words, original_lines):
    """
    Groups a flat list of timed words into segments based on the
    line structure of the original transcript. It also reformats
    the word-level dictionary key from 'text' to 'word'.

    Args:
        timed_words (list): The flat list of word timestamp dicts from the aligner.
        original_lines (list): The list of strings from the original .txt file.

    Returns:
        dict: The final JSON structure with accurately segmented sentences.
    """
    output = {
        "language": "ar",
        "segments": []
    }
    
    word_idx = 0  # This acts as a pointer to our position in timed_words
    
    for line in original_lines:
        line = line.strip()
        if not line:
            continue
            
        words_in_line = line.split()
        num_words = len(words_in_line)
        
        # Take a slice of the timed_words list corresponding to the current line
        current_words_chunk = timed_words[word_idx : word_idx + num_words]
        
        if not current_words_chunk:
            continue

        # Create a new list of word dicts with the correct key "word"
        formatted_words = [
            {
                "start": word.get("start"),
                "end": word.get("end"),
                "word": word.get("text"), # The key is now "word"
                "score": word.get("score")
            } for word in current_words_chunk
        ]

        # Create the segment using the data from the current line's words
        segment = {
            "end": current_words_chunk[-1]['end'],
            "start": current_words_chunk[0]['start'],
            "text": line, # This is the segment-level text
            "words": formatted_words # Use the newly formatted list
        }
        
        output["segments"].append(segment)
        
        # Move the pointer forward for the next line
        word_idx += num_words
        
    return output

def to_srt_time(seconds):
    """Convert float seconds to SRT time format HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(word_timestamps):
    """Generate SRT content from word timestamps"""
    srt_content = ""
    for idx, entry in enumerate(word_timestamps, start=1):
        start_time = to_srt_time(entry['start'])
        end_time = to_srt_time(entry['end'])
        text = entry['text']
        
        srt_content += f"{idx}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{text}\n\n"
    
    return srt_content

def perform_forced_alignment(audio_path, text_content, language="ara", batch_size=16):
    """
    Perform forced alignment on audio and text content.
    
    Args:
        audio_path: Path to audio file
        text_content: Text content as string
        language: Language code (default: "ara")
        batch_size: Batch size for processing (default: 16)
    
    Returns:
        list: word_timestamps
    """
    # Initialize model if not already loaded
    initialize_model()
    
    # Load audio
    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)
    
    # Clean text - preserve line structure for later segmentation
    text = text_content.replace("\n", " ").strip()
    
    # Generate emissions
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )
    
    # Preprocess text
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )
    
    # Get alignments
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )
    
    # Get spans
    spans = get_spans(tokens_starred, segments, blank_token)
    
    # Get word timestamps
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    return word_timestamps

def handler(job):
    """
    Main handler for serverless forced alignment.
    
    Expected input format:
    {
        "audio": "http://example.com/audio.mp3" or local path,
        "text": "http://example.com/text.txt" or text content,
        "language": "ara" (optional, defaults to Arabic),
        "batch_size": 16 (optional),
        "output_format": "both" (optional: "srt", "json", or "both"),
        "segment_by_lines": true (optional, create segments based on text lines)
    }
    
    Returns:
    {
        "srt": "SRT content string" (if requested),
        "json": {...} (if requested),
        "word_timestamps": [...] (always included)
    }
    """
    try:
        # Get input parameters
        input_data = job.get("input", {})
        audio_input = input_data.get("audio")
        text_input = input_data.get("text")
        language = input_data.get("language", "ara")
        batch_size = input_data.get("batch_size", 16)
        output_format = input_data.get("output_format", "both")
        segment_by_lines = input_data.get("segment_by_lines", True)
        
        if not audio_input or not text_input:
            return {"error": "Both 'audio' and 'text' parameters are required"}
        
        # Create temp directory if it doesn't exist
        os.makedirs("./tmp", exist_ok=True)
        
        audio_path = None
        text_content = None
        original_lines = []
        temp_files_to_cleanup = []
        
        try:
            # Handle audio input (URL or local path)
            if audio_input.startswith("http"):
                print("Downloading audio file...")
                audio_path = download_file(audio_input, suffix='.mp3')
                temp_files_to_cleanup.append(audio_path)
            else:
                audio_path = audio_input
            
            # Handle text input (URL, local path, or direct text content)
            if text_input.startswith("http"):
                print("Downloading text file...")
                text_path = download_file(text_input, suffix='.txt')
                temp_files_to_cleanup.append(text_path)
                with open(text_path, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()
                    text_content = "".join(line for line in original_lines).replace("\n", " ").strip()
            elif os.path.exists(text_input):
                # Local file path
                with open(text_input, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()
                    text_content = "".join(line for line in original_lines).replace("\n", " ").strip()
            else:
                # Direct text content
                text_content = text_input
                original_lines = text_input.split('\n')
            
            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found: {audio_path}"}
            
            if not text_content or not text_content.strip():
                return {"error": "Text content is empty or invalid"}
            
            print("Starting forced alignment...")
            
            # Perform forced alignment
            word_timestamps = perform_forced_alignment(
                audio_path, text_content, language, batch_size
            )
            
            result = {
                "word_timestamps": word_timestamps
            }
            
            # Generate outputs based on requested format
            if output_format in ["srt", "both"]:
                print("Generating SRT content...")
                srt_content = generate_srt(word_timestamps)
                result["srt"] = srt_content
                
                # Optionally save SRT file to temp (for debugging)
                try:
                    srt_path = "./tmp/output.srt"
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_content)
                    temp_files_to_cleanup.append(srt_path)
                except:
                    pass  # Ignore file save errors in serverless environment
            
            if output_format in ["json", "both"]:
                print("Generating JSON output...")
                # Create segmented output if original lines are available and requested
                if segment_by_lines and len(original_lines) > 1:
                    # Multi-line text - create segments using the original segmentation logic
                    formatted_segments = create_segments_from_lines(word_timestamps, original_lines)
                    result["json"] = formatted_segments
                    result["segmented"] = True
                else:
                    # Single line or continuous text - return simple structure
                    formatted_output = {
                        "language": language,
                        "word_timestamps": word_timestamps
                    }
                    result["json"] = formatted_output
                    result["segmented"] = False
                
                # Optionally save JSON file to temp (for debugging)
                try:
                    json_path = "./tmp/output_sentence_segmented.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(result["json"], f, indent=4, ensure_ascii=False)
                    temp_files_to_cleanup.append(json_path)
                except:
                    pass  # Ignore file save errors in serverless environment
            
            print("✅ Forced alignment completed successfully")
            return result
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
        
        finally:
            # Cleanup temporary files
            for temp_file in temp_files_to_cleanup:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors
    
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# Initialize model on container startup
try:
    initialize_model()
except Exception as e:
    print(f"Warning: Could not initialize model on startup: {e}")

# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

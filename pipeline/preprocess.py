import re
import os

def preprocess_dialogues(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Split content by "Audio" markers
    # Matches patterns like: Audio 1, Audio2, Audiofile 3, AUIDO 5, AUDIO FILE 6
    split_pattern = r'(?i)\b(?:audio|auido)\s*(?:file)?\s*\d+\b'
    
    # We find all matches to know what to split on
    markers = re.findall(split_pattern, content)
    
    # Split the text
    dialogue_blocks = re.split(split_pattern, content)
    
    # The first block might be "Doctor patient converstation" or empty, skip it if it doesn't contain dialogue
    processed_dialogues = []
    
    for block in dialogue_blocks:
        if not block.strip():
            continue
        
        # Check if this block looks like dialogue or just header text
        if 'Doctor:' not in block and 'Patient:' not in block:
            continue
            
        # 2. Clean up each dialogue block
        lines = block.strip().split('\n')
        cleaned_lines = []
        
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Normalize speaker tags
            # Matches: "Doctor:", "Patient:", "Patient(father):", "*Patient:", "Patient(mother):"
            speaker_match = re.match(r'^[\*]?\b(Doctor|Patient)(?:\([^\)]+\))?\s*:\s*(.*)', line, re.IGNORECASE)
            
            if speaker_match:
                # If we were building a previous turn, save it
                if current_speaker:
                    cleaned_lines.append(f"{current_speaker}: {current_text.strip()}")
                
                current_speaker = speaker_match.group(1).capitalize()
                current_text = speaker_match.group(2)
            else:
                # If it doesn't match a speaker but there's a current speaker, it's a multi-line turn
                if current_speaker:
                    # Clean up stray artifacts like ":" on its own line
                    if line == ":":
                        continue
                    current_text += " " + line
                else:
                    # Misc lines before the first speaker in a block (header trash)
                    continue

        # Add the last turn
        if current_speaker:
            cleaned_lines.append(f"{current_speaker}: {current_text.strip()}")
            
        # 3. Final cleaning of the combined dialogue
        dialogue_str = " ".join(cleaned_lines)
        
        # Remove placeholders like ****
        dialogue_str = re.sub(r'\*+', '', dialogue_str)
        
        # Fix punctuation spacing: Word?Word -> Word? Word
        dialogue_str = re.sub(r'([?.!,])([a-zA-Z])', r'\1 \2', dialogue_str)
        
        # Fix smart quotes and ellipses
        dialogue_str = dialogue_str.replace('’', "'").replace('“', '"').replace('”', '"').replace('…', '...')
        
        # Replace multiple spaces with single space
        dialogue_str = re.sub(r'\s+', ' ', dialogue_str).strip()
        
        if dialogue_str:
            processed_dialogues.append(dialogue_str)

    # 4. Save result
    with open(output_path, 'w', encoding='utf-8') as f:
        for d in processed_dialogues:
            f.write(d + '\n')
            
    print(f"Successfully processed {len(processed_dialogues)} dialogues to {output_path}")

if __name__ == "__main__":
    input_file = "raw_inp.txt"
    output_file = "processed_inputs.txt"
    preprocess_dialogues(input_file, output_file)

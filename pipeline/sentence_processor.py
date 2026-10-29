"""
Sentence Processor: Parses dialogue files and splits into sentences
Handles medical dialogue format with speaker tags (Doctor, Patient)
"""

import re
from typing import List, Dict


class SentenceProcessor:
    """Process dialogues and extract sentences while preserving speaker tags"""

    @staticmethod
    def parse_dialogue_file(file_path: str) -> List[Dict]:
        """
        Parse dialogue from file preserving speaker tags and sentences.

        Args:
            file_path: Path to dialogue file

        Returns:
            List of dicts with 'speaker' and 'sentence'
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return SentenceProcessor.parse_dialogue_content(content)

    @staticmethod
    def parse_dialogue_content(text: str) -> List[Dict]:
        """
        Parse raw dialogue text preserving speaker and splitting into sentences.
        """
        pattern = r'(?i)\b(Doctor|Patient|Nurse)(?:\([^\)]+\))?\s*:\s*'
        splits = re.split(pattern, text)
        
        results = []
        if len(splits) >= 3:
            for i in range(1, len(splits), 2):
                speaker = splits[i].strip().capitalize()
                turn_text = splits[i + 1].strip()
                sub_sentences = SentenceProcessor.split_sentences(turn_text)
                for sub_s in sub_sentences:
                    s_clean = sub_s.strip()
                    if s_clean:
                        results.append({
                            'speaker': speaker,
                            'sentence': s_clean
                        })
        else:
            # Fallback if no speaker tags present
            sub_sentences = SentenceProcessor.split_sentences(text)
            for sub_s in sub_sentences:
                s_clean = sub_s.strip()
                if s_clean:
                    results.append({
                        'speaker': 'Unknown',
                        'sentence': s_clean
                    })
                    
        return results

    @staticmethod
    def split_by_speakers(text: str) -> List[str]:
        """
        Split text by speaker tags (backward compatibility)
        """
        pattern = r'(?:Doctor|Patient|Nurse):\s*'
        statements = re.split(pattern, text)
        return [s.strip() for s in statements if s.strip()]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into individual sentences by punctuation
        Preserves sentence boundaries

        Args:
            text: Statement text (may contain multiple sentences)

        Returns:
            List of sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    @staticmethod
    def create_sentence_metadata(sentences: List) -> List[Dict]:
        """
        Create metadata for each sentence (ID, sentence content, speaker)

        Args:
            sentences: List of sentence strings or dicts with 'speaker'/'sentence'

        Returns:
            List of dicts with 'id', 'sentence', and 'speaker' keys
        """
        metadata = []
        for idx, item in enumerate(sentences, 1):
            if isinstance(item, dict):
                metadata.append({
                    'id': f'sent_{idx:04d}',
                    'sentence': item.get('sentence', ''),
                    'speaker': item.get('speaker', 'Unknown')
                })
            else:
                metadata.append({
                    'id': f'sent_{idx:04d}',
                    'sentence': str(item),
                    'speaker': 'Unknown'
                })
        return metadata


def load_dialogues(file_path: str) -> List[Dict]:
    """
    Load dialogue file and return sentences with metadata (including speaker)

    Args:
        file_path: Path to dialogue file

    Returns:
        List of dicts with 'id', 'sentence', and 'speaker'
    """
    processor = SentenceProcessor()
    parsed_items = processor.parse_dialogue_file(file_path)
    metadata = processor.create_sentence_metadata(parsed_items)
    return metadata

"""
Quick check if specific words exist in the tokenizer vocabulary
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tokenizer.tokenizer import Tokenizer

# Load the trained tokenizer
checkpoint_path = Path(__file__).parent.parent.parent / 'tokenizer/trained/owt_train/final_0032000_inference.pkl'
print("Loading tokenizer...")
tokenizer = Tokenizer.load_for_inference(checkpoint_path)
print(f"Loaded tokenizer with vocab size: {tokenizer.token_registery.num_tokens}\n")

# Words to check
words_to_check = [
    "is",
    "hello", 
    "Hello",
    "the",
    "and",
    "world",
    "test",
    " is",      # with space
    " hello",  # with space
    " the",
    " and",
]

print("=" * 80)
print("CHECKING IF WORDS EXIST AS SINGLE TOKENS IN VOCABULARY")
print("=" * 80)

for word in words_to_check:
    word_bytes = word.encode('utf-8')
    token = tokenizer.token_registery.get_token_by_bytes(word_bytes)
    
    if token:
        print(f"✓ '{word}' EXISTS as token #{token.token_idx} (freq: {token.token_freq:,})")
    else:
        print(f"✗ '{word}' does NOT exist as a single token")

print("\nDone!")

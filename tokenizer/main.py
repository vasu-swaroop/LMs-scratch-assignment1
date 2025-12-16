import sys
import os

# Ensure the current directory is in python path to allow imports from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizer import Tokenizer, TokenizerConfig
from pathlib import Path

def main():
    print("Initializing Tokenizer...")
    # Calculate absolute path to data file
    # Assuming main.py is in tokenizer/ and data is in ../data/
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'data' / 'owt_valid.txt'
    
    config = TokenizerConfig(corpus_path=data_path, vocab_size=300)
    tokenizer = Tokenizer(config=config)
    
    print(f"Using corpus path: {data_path}")
    print("Starting Tokenizer Training...")
    tokenizer.train_tokenizer()
    print("Tokenizer Training Completed.")

if __name__ == "__main__":
    main()

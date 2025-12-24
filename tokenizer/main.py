import sys
import os
from .tokenizer import Tokenizer, TokenizerConfig
from pathlib import Path

def main():
    print("Initializing Tokenizer...")
    # Calculate absolute path to data file
    # Assuming main.py is in tokenizer/ and data is in ../data/
    config = TokenizerConfig()
    tokenizer = Tokenizer(config=config)
    
    print("Starting Tokenizer Training...")
    tokenizer.train_tokenizer()
    print(tokenizer.print_sample_results())
    print("Tokenizer Training Completed.")

if __name__ == "__main__":
    main()

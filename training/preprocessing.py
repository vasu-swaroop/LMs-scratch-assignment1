from tokenizer.tokenizer import Tokenizer
from tokenizer.utils import find_chunk_boundaries
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import os

# Configuration
tokenizer_path = '/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl'
root_data_path = '/data3/vasu/projects/LMs-scratch-assignment1/data/TinyStoriesV2-GPT4-valid.txt'
data_set_save_path = Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/TinyStoriesV2_valid/')
num_shards = 10

# Load tokenizer
tokenizer = Tokenizer.load_for_inference(tokenizer_path)

# Ensure save directory exists
data_set_save_path.mkdir(exist_ok=True, parents=True)

def process_chunk(start: int, end: int):
    with open(root_data_path, 'rb') as f:
        f.seek(start)
        # Read the chunk
        chunk_bytes = f.read(end - start)
        # Decode and handle errors
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
        # Tokenize (returns a list of integers)
        token_list = tokenizer.inference_on_text(chunk_text,ret_type='int')
        
    # Convert to a compact numpy array (uint16/uint32 depending on vocab size)
    # Vocab size is 32000, so uint16 (up to 65535) is sufficient
    toks=[tokens for words in token_list for tokens in words]
    tokens_np = np.array(toks, dtype=np.uint16)
    
    save_path = data_set_save_path / f'{start}_{end}.data'

    # Create memmap with actual data length and write tokens
    mm = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=tokens_np.shape)
    mm[:] = tokens_np[:]
    mm.flush()

def main():
    # Find chunk boundaries
    with open(root_data_path, 'rb') as file:
        boundaries = find_chunk_boundaries(file, num_shards, "<|endoftext|>".encode())
    print(boundaries)
    # Process in parallel
    print(f"Starting parallel processing of {len(boundaries)} chunks...")
    Parallel(n_jobs=-1)(
        delayed(process_chunk)(start, end) for start, end in zip(boundaries[:-1],boundaries[1:])
    )
    print("Processing complete.")

if __name__ == "__main__":
    main()
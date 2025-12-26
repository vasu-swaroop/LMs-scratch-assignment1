import numpy as np
from pathlib import Path
import random
from joblib import delayed, Parallel
from jax import numpy as jnp

class Document():
    def __init__(self, document_path):
        self.np_array = np.memmap(
            document_path,
            dtype=np.uint16,
            mode="r",
        )
        self.cur_pos = 0

    def sample_text(self, seq_len):
        out_arr = self.np_array[self.cur_pos:self.cur_pos+seq_len]
        # Randomly advance or slightly backtrack to vary the sequence start
        self.cur_pos += 1
        
        # If we reach the end, reset or pad
        if len(out_arr) < seq_len:
            # Simple padding with zeros for short end-of-file chunks
            padding = np.zeros(seq_len - len(out_arr), dtype=np.uint16)
            out_arr = np.concatenate([out_arr, padding])
            self.cur_pos = 0 # Reset to start for next sample
        return out_arr

def sample_text(doc: Document, seq_len: int):
    return doc.sample_text(seq_len)

class Data():
    def __init__(self, root_path: Path, batch_size: int, steps: int, seq_len: int):
        # Only read files with .data extension to avoid hidden files
        all_docs_path = [p for p in root_path.iterdir() if p.suffix == '.data']
        self.documents = [Document(doc) for doc in all_docs_path]        
        self.total_docs_idx = range(len(self.documents))
        self.steps = steps
        self.batch_size = batch_size
        self.seq_len = seq_len

    def data_loader(self):
        # Use threading backend so Document.cur_pos state is shared across workers
        parallel_sequencer = Parallel(n_jobs=-1, backend="threading")
        for step in range(self.steps):
            # We need at least batch_size+1 samples to create input/output pairs
            num_docs_needed = self.batch_size + 1
            
            if len(self.total_docs_idx) >= num_docs_needed:
                # Enough documents - sample without replacement
                chosen_docs = random.sample(self.total_docs_idx, num_docs_needed)
            else:
                # Not enough documents - sample with replacement
                chosen_docs = random.choices(list(self.total_docs_idx), k=num_docs_needed)
            
            results = parallel_sequencer(
                delayed(sample_text)(self.documents[idx], self.seq_len) for idx in chosen_docs
            )
            results=jnp.array(results)
            yield results[:-1], results[1:]

if __name__ == '__main__':
    in_ = Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/overfiting_test')
    data = Data(in_, 10, 100, 103)
    for x in data.data_loader():
        print(x)

from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Union  
from jax import numpy as jnp
import numpy as np
from .pre_tokenization import PreToken, PreTokenRegistry
from .tokens import Token, TokenRegistery
from .token_pair import TokenPairRegistry
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import joblib
import regex as re

@dataclass
class TokenizerConfig:
    vocab_size: int= 32_000  
    corpus_path: Path= Path('/data3/vasu/projects/LMs-scratch-assignment1/data/owt_train.txt')
    tie_resolution: str= 'lexicographic'
    token_dict: dict[int,bytes]= field(default_factory=dict)
    train_steps: int =0 
    pretokeinzed_corpus_path: str= Path('owt_train_pretokens.pkl') 
    separating_token: list[str] = field(default_factory=lambda: ['<|endoftext|>'])
    stats_file: Path = Path('tokenizer_timing_stats.txt')
    save_freq: int = 10_000
    resume_from_checkpoint: bool = False
    save_file_dir: Path= Path('tokenizer/trained/owt_train')

class Tokenizer():
    def __init__(self, config: TokenizerConfig = None):
        self.config: TokenizerConfig= config if config is not None else TokenizerConfig()
        self.token_pair_registry: TokenPairRegistry= TokenPairRegistry()
        self.token_registery: TokenRegistery= TokenRegistery()
        self.pre_token_registery: PreTokenRegistry= PreTokenRegistry(self.config.corpus_path, self.config.separating_token)

    def run_pretokenization(self,save_path, num_processes=4):
        self.pre_token_registery.populate_pre_token(num_processes)
        self.pre_token_registery.remove_sep_pattern() # We remove in pre_token_reg such that it doesn't get considered for the token_pair update
        self.pre_token_registery.save_pre_token(save_path)

    def observe_token_pair(self, token_A:Token, token_B:Token, pre_token: PreToken, count_pre_token: int):
        self.token_pair_registry.observe(token_A, token_B, pre_token, count_pre_token)
    
    def initialize_token_pair(self):
        for pre_token in tqdm(self.pre_token_registery.list_pre_tokens(), desc="Initializing Token Pairs"):
            count_pre_token=pre_token.freq
            token_arr=pre_token.token_arr
            for token_idx in range(len(token_arr)-1):
                token_A,token_B= token_arr[token_idx], token_arr[token_idx+1]
                self.token_pair_registry.observe(token_A, token_B, pre_token, count_pre_token)

    
    def _unobserve_if_exists(self, A: Token, B: Token, pre_token: PreToken, freq: int, deleted_A: Token, deleted_B: Token):
        """Safely un-observes a pair, checking existence and avoiding the deleted pair."""
        if not A or not B:
            return
            
        # Optimization: Don't check for the pair we know we just deleted
        if A == deleted_A and B == deleted_B:
            return

        # Safety: Only un-observe if it's actually in the registry
        if (A, B) in self.token_pair_registry._pairs:
            self.token_pair_registry.un_observe(A, B, pre_token, freq)

    def update_the_pair_post_merge(self, token_A:Token, token_B:Token, following:Token, preceeding:Token, pre_token: PreToken, pre_token_freq: int, new_token: Token):
        # Un-observe neighbors safely
        self._unobserve_if_exists(preceeding, token_A, pre_token, pre_token_freq, token_A, token_B)
        self._unobserve_if_exists(token_B, following, pre_token, pre_token_freq, token_A, token_B)

        # Always observe the new pairs formed with the new merged token
        self.token_pair_registry.observe(preceeding, new_token, pre_token, pre_token_freq)
        self.token_pair_registry.observe(new_token, following, pre_token, pre_token_freq)

    def update_pre_tokens(self,pre_token:PreToken, new_token:Token, token_A:Token, token_B:Token, ):
        preceeding, following= pre_token.modify_pre_token_rep(new_token, token_A, token_B)    
        pre_token_freq=pre_token.get_freq()
        # print("new_token",new_token,"merging", token_A, token_B, "with f as",following, "with count as ",pre_token_freq)
        # print()
        # Add if the preceding and following not in the pre_token anymore post merging, remove pre_token from the token_pair list
        self.update_the_pair_post_merge(token_A, token_B, following, preceeding, pre_token, pre_token_freq, new_token)

    def merge_most_frequent_token_pair(self):
        token_pair_metadata=self.token_pair_registry.get_most_frequent_token_pair()
        
        token_pair_list=token_pair_metadata.get_pre_token_list()
        token_A, token_B= token_pair_metadata.get_token()

        #delete the token pair
        self.token_pair_registry.delete_pair(token_A, token_B)

        # Create a new token
        merged_bytes=token_A.byte_arr+token_B.byte_arr
        merged_tokens=(token_A, token_B)
        new_token= self.token_registery.add_tokens(merged_bytes, token_pair_metadata.token_pair_count,merged_tokens) 
        
        # Update the PreTokenFreq list of pretokens using pretoken registery        
        for pre_tokens in token_pair_list: 
            self.update_pre_tokens(pre_tokens, new_token, token_A, token_B)    
        # Since the update step is neither CPU bound/Io bound, we can let it be serial.
 
    def print_sample_results(self):
        from pprint import pprint as pp
        pp(self.token_registery._tokens)

    def save_tokenizer_state(self, step:int,save_name:str):
        """Save full tokenizer state for resuming training (large file)."""
        base=self.config.save_file_dir
        base.mkdir(exist_ok=True, parents=True)
        save_path=base /f'{save_name}_{step:7d}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    def save_for_inference(self, step:int, save_name:str='final'):
        """Save only what's needed for inference (small file ~1MB).
        
        Only saves token_registery which contains the vocabulary.
        This is much smaller than saving the full tokenizer state.
        """
        base=self.config.save_file_dir
        base.mkdir(exist_ok=True, parents=True)
        save_path=base / f'{save_name}_{step:07d}_inference.pkl'
        
        # Only save what's needed for inference
        inference_data = {
            'token_registery': self.token_registery,
            'config': self.config,
            'vocab_size': step,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(inference_data, f)
        
        return save_path

    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """Load a tokenizer from a full checkpoint (running_*.pkl).
        
        Returns the full tokenizer with all registries for continued training or inference.
        """
        with open(checkpoint_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        return tokenizer

    @staticmethod
    def load_for_inference(checkpoint_path):
        """Load a tokenizer from an inference checkpoint (*_inference.pkl).
        
        Returns a minimal object with token_registery for encoding/decoding.
        """
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if this is a full checkpoint instead of inference checkpoint
        if isinstance(data, Tokenizer):
            # This is a running checkpoint, not an inference checkpoint
            return data
        
        # Create a minimal tokenizer instance
        tokenizer = Tokenizer(config=data['config'])
        tokenizer.token_registery = data['token_registery']
        
        return tokenizer

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint file if it exists."""
        base = self.config.save_file_dir
        if not base.exists():
            return None, 0
        
        # Look for running checkpoints
        checkpoints = list(base.glob('running_*.pkl'))
        if not checkpoints:
            return None, 0
        
        # Extract vocab sizes and find the latest
        checkpoint_info = []
        for ckpt in checkpoints:
            try:
                # Extract the vocab size from filename: running_0001000.pkl -> 1000
                vocab_size = int(ckpt.stem.split('_')[1])
                checkpoint_info.append((vocab_size, ckpt))
            except (ValueError, IndexError):
                continue
        
        if not checkpoint_info:
            return None, 0
        
        # Return the checkpoint with the highest vocab size
        checkpoint_info.sort(reverse=True)
        vocab_size, checkpoint_path = checkpoint_info[0]
        return checkpoint_path, vocab_size

        
    def train_tokenizer(self):
        import time
        stats = []
        
        # Check for existing checkpoint to resume from
        checkpoint_path, resumed_vocab_size = None, 0
        if self.config.resume_from_checkpoint:
            checkpoint_path, resumed_vocab_size = self.find_latest_checkpoint()
            
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path} (vocab_size={resumed_vocab_size})")
            start_time = time.time()
            with open(checkpoint_path, 'rb') as f:
                loaded_tokenizer = pickle.load(f)
            # Restore the state from checkpoint
            self.token_registery = loaded_tokenizer.token_registery
            self.token_pair_registry = loaded_tokenizer.token_pair_registry
            self.pre_token_registery = loaded_tokenizer.pre_token_registery
            load_time = time.time() - start_time
            stats.append(f"Checkpoint loaded: {load_time:.2f}s")
            print(f"Resumed training from vocab_size={resumed_vocab_size}")
        else:
            # Start from scratch - do pretokenization and initialization
            start_time = time.time()
            print("Starting Pre-tokenization...")
            save_tokenization_path=self.config.save_file_dir / self.config.pretokeinzed_corpus_path
            
            # Check if pretokenized corpus already exists
            if save_tokenization_path.exists():
                print(f"Loading existing pretokenized corpus from {save_tokenization_path}...")
                with open(save_tokenization_path, 'rb') as f:
                    self.pre_token_registery = pickle.load(f)
                pretok_time = time.time() - start_time
                stats.append(f"Pre-tokenization (loaded): {pretok_time:.2f}s")
            else:
                self.run_pretokenization(save_tokenization_path)
                pretok_time = time.time() - start_time
                stats.append(f"Pre-tokenization: {pretok_time:.2f}s")
            
            # Calculate base token frequencies
            start_time = time.time()
            print("Calculating Base Frequencies...")
            counter = Counter()
            for pre_token in tqdm(self.pre_token_registery.list_pre_tokens(), desc="Calculating Base Frequencies"):
                 for token in pre_token.token_arr:
                     # token is a base token (freq=None)
                     counter[token.token_idx] += pre_token.freq
            freq_time = time.time() - start_time
            stats.append(f"Base Frequency Calculation: {freq_time:.2f}s")
                     
            start_time = time.time()
            print("Initializing Token Pairs...")
            self.token_registery.default_init(counter=counter, special_tokens=self.config.separating_token)
            self.initialize_token_pair()
            init_pair_time = time.time() - start_time
            stats.append(f"Token Pair Initialization: {init_pair_time:.2f}s")
        

        print("Starting Merge Loop...")
        start_time = time.time()
        
        current_vocab_size=self.token_registery.num_tokens
        pbar = tqdm(total=self.config.vocab_size - current_vocab_size)
        while current_vocab_size<self.config.vocab_size:
            self.merge_most_frequent_token_pair()
            current_vocab_size=self.token_registery.num_tokens
            if current_vocab_size%self.config.save_freq==0:
                self.save_tokenizer_state(current_vocab_size, 'running')

            pbar.update(1)
            pbar.set_description(f"Vocab size: {current_vocab_size}")
        pbar.close()
        merge_time = time.time() - start_time
        stats.append(f"Merge Loop: {merge_time:.2f}s")

        start_time = time.time()
        print("Saving Tokenizer...")
        # Save lightweight inference model
        inference_path = self.save_for_inference(current_vocab_size)
        print(f"Saved inference model to: {inference_path}")
        save_time = time.time() - start_time
        stats.append(f"Saving Tokenizer (inference): {save_time:.2f}s")
        
        # Save stats to file
        save_file_name=self.config.save_file_dir/self.config.stats_file
        with open(save_file_name, 'w') as f:
            f.write("\n".join(stats))
        print(f"Timing stats saved to {self.config.stats_file}")

    def split_pretokens(self, text):
        regex = self.pre_token_registery.regex_split_pattern
        words = re.findall(regex, text)
        return words

    def tokenize(self, token_stream:List[Token])-> List[Token]:
        token_present=self.token_registery.get_token_by_bytes(b''.join([token.byte_arr for token in token_stream]))
        if token_present:
            return [token_present ]
        cur_token_stream = token_stream
        merge_pending=True
        
        while(merge_pending):
            new_token_stream=[]
            new_token_stream.append(cur_token_stream[0])

            for token in cur_token_stream[1:]:
                # Check if concatenated bytes exist as a token in vocabulary
                merged_bytes = new_token_stream[-1].byte_arr + token.byte_arr
                merged_token = self.token_registery.get_token_by_bytes(merged_bytes)
                
                if merged_token is not None:
                    new_token_stream[-1] = merged_token
                else:
                    new_token_stream.append(token)

            merge_pending= not(len(new_token_stream)==len(cur_token_stream))
            cur_token_stream=new_token_stream

        return new_token_stream

    def inference_on_text(self, text: str, ret_type: str = 'tokens') -> Union[List[List[Token]], List[List[int]]]:
        """Tokenize text and return either token objects or token indices.
        
        Args:
            text: Input text to tokenize
            ret_type: 'int' to return token indices, 'tokens' to return Token objects
            
        Returns:
            List of token lists (either Token objects or integers)
        """
        # Split text into pretokens
        all_pretokens = self.split_pretokens(text)
        
        # Convert pretokens (bytes) to initial Token objects
        all_pretokens_tokens = []
        for pretoken in all_pretokens:
            # Convert each byte in the pretoken to a Token using efficient byte lookup
            token_list = [self.token_registery.get_token_by_bytes(bytes([byte])) for byte in pretoken.encode('utf-8')]
            all_pretokens_tokens.append(token_list)
        
        # Apply BPE merging to each pretoken's token list
        all_words_tokens = Parallel(n_jobs=1)(delayed(self.tokenize)(token_list) for token_list in all_pretokens_tokens)

        if ret_type == 'int':
            return [[token.token_idx for token in token_list] for token_list in all_words_tokens]
        else:
            return all_words_tokens

    def tokens_to_text(self, input_data: list[int]|list[Token]):
        if isinstance(input_data[0], jnp.ndarray):
            input_data = np.asarray(input_data.astype(int)).flatten()
            input_data = [self.token_registery.get_token(t).byte_arr for t in input_data]
        elif isinstance(input_data[0],int):  # noqa: E721
            input_data = [self.token_registery.get_token(t).byte_arr for t in input_data]
        else:
            input_data = [t.byte_arr for t in input_data]
        decoded = b''.join(input_data)
        decoded = decoded.decode('utf-8', errors='replace')
        
        return decoded

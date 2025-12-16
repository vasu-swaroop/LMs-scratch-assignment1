
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from pre_tokenization import PreToken, PreTokenRegistry
from tokens import Token, TokenRegistery
from token_pair import TokenPairRegistry

@dataclass
class TokenizerConfig:
    vocab_size: int= 65536  
    corpus_path: Path= Path('../data/owt_train.txt')
    tie_resolution: str= 'lexicographic'
    token_dict: dict[int,bytes]= field(default_factory=dict)
    train_steps: int =0 
    pretokeinzed_corpus_path: Path= Path('../data/owt_train_pretokenized.pkl') 

class Tokenizer():
    def __init__(self, config: TokenizerConfig = None):
        self.config: TokenizerConfig= config if config is not None else TokenizerConfig()
        self.token_pair_registry: TokenPairRegistry= TokenPairRegistry()
        self.token_registery: TokenRegistery= TokenRegistery()
        self.pre_token_registery: PreTokenRegistry= PreTokenRegistry(self.config.corpus_path)
    def run_pretokenization(self, num_processes=4):
        self.pre_token_registery.populate_pre_token(num_processes)
        #TODO: Add caching mechanism

    def observe_token_pair(self, token_A:Token, token_B:Token, pre_token: PreToken, count_pre_token: int):
        self.token_pair_registry.observe(token_A, token_B, pre_token, count_pre_token)
    
    def initialize_token_pair(self):
        from tqdm import tqdm
        for pre_token in tqdm(self.pre_token_registery.list_pre_tokens(), desc="Initializing Token Pairs"):
            count_pre_token=pre_token.freq
            token_arr=pre_token.token_arr
            for token_idx in range(len(token_arr)-1):
                token_A,token_B= token_arr[token_idx], token_arr[token_idx+1]
                self.token_pair_registry.observe(token_A, token_B, pre_token, count_pre_token)

    def merge_most_frequent_token_pair(self):
        token_pair_metadata=self.token_pair_registry.get_most_frequent_token_pair()
        
        token_pair_list=token_pair_metadata.get_pre_token_list()
        token_A, token_B= token_pair_metadata.get_token()

        #delete the token pair
        self.token_pair_registry.delete_pair(token_A, token_B)

        # Create a new token
        merged_bytes=token_A.byte_arr+token_B.byte_arr
        new_token= self.token_registery.add_tokens(merged_bytes, token_pair_metadata.token_pair_count) 

        # Update the PreTokenFreq list of pretokens using pretoken registery
        for pre_token in token_pair_list:
            preceeding, following= pre_token.modify_pre_token_rep(new_token, token_A, token_B)    
            pre_token_freq=pre_token.get_freq()
            # print("new_token",new_token,"merging", token_A, token_B, "with f as",following, "with count as ",pre_token_freq)
            # print()

            # Add if the preceding and following not in the pre_token anymore post merging, remove pre_token from the token_pair list
            self.token_pair_registry.un_observe(preceeding, token_A,pre_token, pre_token_freq)
            self.token_pair_registry.un_observe(token_B, following,pre_token, pre_token_freq)
            self.token_pair_registry.observe(preceeding, new_token,pre_token, pre_token_freq)
            self.token_pair_registry.observe(new_token, following,pre_token, pre_token_freq)
    def save_tokenizer(self):
        import pickle
        with open("token_pair.pkl", "wb") as f:
            pickle.dump(self.token_registery, f)
        
    def print_sample_results(self):
        from pprint import pprint as pp
        pp(self.token_registery._tokens)
    def train_tokenizer(self):
        self.run_pretokenization()
        
        # Calculate base token frequencies
        counter = Counter()
        from tqdm import tqdm
        for pre_token in tqdm(self.pre_token_registery.list_pre_tokens(), desc="Calculating Base Frequencies"):
             for token in pre_token.token_arr:
                 # token is a base token (freq=None)
                 counter[token.token_idx] += pre_token.freq
                 
        self.token_registery.default_init(counter=counter)
        self.initialize_token_pair()
        current_vocab_size=self.token_registery.num_tokens
        from tqdm import tqdm
        pbar = tqdm(total=self.config.vocab_size - current_vocab_size)
        while current_vocab_size<self.config.vocab_size:
            self.merge_most_frequent_token_pair()
            current_vocab_size=self.token_registery.num_tokens
            pbar.update(1)
            pbar.set_description(f"Vocab size: {current_vocab_size}")
        pbar.close()
        self.save_tokenizer()
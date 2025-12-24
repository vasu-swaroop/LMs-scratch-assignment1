import os
import regex as re
import pickle
from dataclasses import dataclass, field
from typing import BinaryIO
from joblib import Parallel, delayed
from threading import Lock
from .tokens import Token
from collections import Counter
from .utils import find_chunk_boundaries

def process_chunk(corpus_path: str, start: int, end: int, regex_pattern: str) -> Counter:
    counts = Counter()
    with open(corpus_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        matches = re.finditer(regex_pattern, chunk)
        for match in matches:
            token_str = match.group()
            counts[token_str] += 1
    return counts

@dataclass(unsafe_hash=True, slots=True)
class PreToken:
    pre_token_idx: int
    token_arr: list[Token] = field(compare=False, hash=False) #TODO: model like a linked list

    freq: int = field(compare=False, hash=False)

    def find_pair(self,token_A:Token, token_B:Token ):
        for token_idx in range(len(self.token_arr)-1):
            if self.token_arr[token_idx]==token_A and self.token_arr[token_idx+1]==token_B:
                return True

        return False


    def modify_pre_token_rep(self, new_token:Token, token_A:Token, token_B:Token)->tuple[Token, Token]:
        preceeding_token = following_token = None
        for token_idx in range(len(self.token_arr)-1):
            if self.token_arr[token_idx]==token_A and self.token_arr[token_idx+1]==token_B:
                if token_idx-1>=0:
                    preceeding_token=self.token_arr[token_idx-1]
                
                if token_idx+2<len(self.token_arr):
                    following_token=self.token_arr[token_idx+2]
                
                self.token_arr[token_idx]=new_token
                self.token_arr[token_idx+1:]=self.token_arr[token_idx+2:]
                return preceeding_token, following_token

        assert True==False, "Error"
    
    def get_freq(self)->int:
        return self.freq


class PreTokenRegistry():
    def __init__(self, corpus_path, separating_tokens=['<|endoftext|']):
        self.corpus_path = corpus_path
        self.pre_token_freq_dict: dict[str, PreToken] = {}
        self.separating_tokens=separating_tokens

        escaped = [re.escape(tok) for tok in self.separating_tokens]
        sep_pattern = "|".join(escaped)
        print(sep_pattern)

        parts = [
            sep_pattern,
            r"'(?:[sdmt]|ll|ve|re)",
            r" ?\p{L}+",
            r" ?\p{N}+",
            r" ?[^\s\p{L}\p{N}]+",
            r"\s+(?!\S)",
            r"\s+"
        ]
        self.regex_split_pattern = "|".join(parts)

    def populate_pre_token(self, num_process=4, num_chunks=2000):
        with open(self.corpus_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        from tqdm import tqdm
        # Map step: Parallel processing of chunks
        results = Parallel(n_jobs=num_process, backend='multiprocessing')(
            delayed(process_chunk)(self.corpus_path, start, end, self.regex_split_pattern)
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])), desc="Pre-tokenization")
        )

        # Reduce step: Aggregate counts
        global_counter = Counter()
        for res in tqdm(results, desc="Aggregating Counts"):
            global_counter.update(res)

        # Create PreToken entries sequentially
        for text, freq in tqdm(global_counter.items(), desc="Creating PreTokens"):
            if text not in self.pre_token_freq_dict:
                self.create_pre_token_entry(text)
            self.pre_token_freq_dict[text].freq = freq
        #TODO: This will add time overhead, remove that, and add in test
        assert all([x in global_counter.keys() for x  in self.separating_tokens]) , "Wrong separating token added"
    def remove_sep_pattern(self):
        for text in self.separating_tokens:
            self.pre_token_freq_dict.pop(text) 

    def save_pre_token(self, save_path):
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def create_pre_token_entry(self, text: str):
        byte_sequence = text.encode()

        if text in self.pre_token_freq_dict:
            return
        pre_token_idx = len(self.pre_token_freq_dict)
        token_list=[]
        for tok in byte_sequence:
            token_list.append(Token(token_idx=int(tok), byte_arr=bytes([tok]), token_freq=None))

        self.pre_token_freq_dict[text] = PreToken(
            pre_token_idx=pre_token_idx,
            token_arr=token_list,
            freq=0
        )

    def list_pre_tokens(self)->list[PreToken]:
        return self.pre_token_freq_dict.values()

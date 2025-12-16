import os
import regex as re
from dataclasses import dataclass, field
from typing import BinaryIO
from joblib import Parallel, delayed
from threading import Lock
from tokens import Token
from utils import find_chunk_boundaries
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
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.pre_token_freq_dict: dict[str, PreToken] = {}
        self.regex_split_pattern:str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._lock = Lock()
    def populate_pre_token(self, num_process=4, num_chunks=2000):
        with open(self.corpus_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        from tqdm import tqdm
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(self.split_on_regex)(start, end)
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])), desc="Pre-tokenization")
        )

    def create_pre_token_entry(self, text: str):
        byte_sequence = text.encode()

        with self._lock:
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

    def split_on_regex(self, start, end)->list[str]:
        with open(self.corpus_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            matches = re.finditer(self.regex_split_pattern, chunk)
            for match in matches:
                token_str=match.group()
                if token_str not in self.pre_token_freq_dict:
                    self.create_pre_token_entry(token_str)
                self.pre_token_freq_dict[token_str].freq+=1

    def list_pre_tokens(self)->list[PreToken]:
        return self.pre_token_freq_dict.values()

        
    

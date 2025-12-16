from dataclasses import dataclass, field
import enum 

from pathlib import Path
from collections import Counter
@dataclass(unsafe_hash=True, slots=True)
class Token:
    token_idx: int
    byte_arr: bytes
    token_freq: int = field(compare=False, hash=False)


class TokenRegistery():
    def __init__(self):
        self._tokens: dict[int, Token] = {}
        self.num_tokens: int = 0
        
    def default_init(self, counter:Counter):
        for i in range(256):
            self._tokens[i]=Token(token_idx=i, byte_arr=bytes([i]),token_freq=counter[i])
        self.num_tokens=256
        
    def add_tokens(self, token_bytes:bytes, token_freq:int)->Token:
        self._tokens[self.num_tokens]=Token(token_idx=self.num_tokens, byte_arr=token_bytes, token_freq=token_freq)
        self.num_tokens+=1
        return self._tokens[self.num_tokens-1]

    def get_token(self, token_idx:int)->Token:
        return self._tokens[token_idx]


    
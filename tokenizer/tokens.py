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
        self._byte_to_token: dict[bytes, Token] = {}  # Reverse lookup: bytes -> Token
        self.num_tokens: int = 0
        self.merge_history: dict[tuple[Token, Token], int] = {}
    def default_init(self, counter: Counter, special_tokens: list[str]):
        for i in range(256):
            token = Token(token_idx=i, byte_arr=bytes([i]), token_freq=counter[i])
            self._tokens[i] = token
            self._byte_to_token[bytes([i])] = token
            
        for token_idx, token_str in enumerate(special_tokens):
            token = Token(token_idx=token_idx+256, byte_arr=token_str.encode(), token_freq=0)
            self._tokens[token_idx+256] = token
            self._byte_to_token[token_str.encode()] = token

        self.num_tokens = len(self._tokens)
        
    def add_tokens(self, token_bytes: bytes, token_freq: int, merge_tokens: tuple[Token, Token]) -> Token:
        new_token = Token(token_idx=self.num_tokens, byte_arr=token_bytes, token_freq=token_freq)
        self._tokens[self.num_tokens] = new_token
        self._byte_to_token[token_bytes] = new_token  # Update reverse lookup
        self.merge_history[merge_tokens] = self.num_tokens
        self.num_tokens += 1
        return new_token

    def get_token(self, token_idx: int) -> Token:
        """Get token by its index."""
        return self._tokens[token_idx]
    
    def get_token_by_bytes(self, byte_arr: bytes) -> Token:
        """Get token by its byte array. O(1) lookup."""
        return self._byte_to_token.get(byte_arr, None)
    
    def is_mergable(self, merge_tokens: tuple[Token, Token]):
        return self.merge_history.get(merge_tokens, None)

    def get_token_dict(self) -> dict[int, Token]:
        return self._tokens


    
from dataclasses import dataclass, field
from tokens import Token
from pre_tokenization import PreToken

@dataclass
class TokenPairMetadata:
    token_A: Token
    token_B: Token
    pre_token_list: set[PreToken]= field(default_factory=set)
    token_pair_count: int=0
    
    def get_token(self)->tuple[Token, Token]:
        return self.token_A, self.token_B

    def get_pre_token_list(self)->set[PreToken]:
        return self.pre_token_list

    def observe(self, pre_token: PreToken, count:int):
        self.token_pair_count += count
        self.pre_token_list.add(pre_token)
        
    def reduce_count(self, count):
        self.token_pair_count -= count


class TokenPairRegistry:
    def __init__(self):
        self._pairs: dict[tuple[Token, Token], TokenPairMetadata] = {}

    def observe(self, A:Token, B:Token, pre_token:PreToken, count:int):
        if not A or not B:
            return 
        key = (A, B)

        if key not in self._pairs:
            self._pairs[key] = TokenPairMetadata(token_A=A, token_B=B)

        self._pairs[key].observe(pre_token, count)

    def un_observe(self, A:Token, B:Token, pre_token:PreToken, count:int):
        if not A or not B:
            return 
        key = (A, B)

        assert key in self._pairs, "Wrong implementation" #Jsut to debug
        
        self._pairs[key].reduce_count(count)
        if not pre_token.find_pair(*key) and pre_token in self._pairs[key].pre_token_list:
            self._pairs[key].pre_token_list.remove(pre_token)

        if self._pairs[key].token_pair_count==0:
            # print(key, self._pairs[key].token_pair_count)
            del self._pairs[key]

    def get_most_frequent_token_pair(self)->TokenPairMetadata:
        token_pair_metadata=max(self._pairs.values(), key=lambda x:x.token_pair_count)
        return token_pair_metadata
    
    def delete_pair(self, token_A, token_B):
        key=(token_A, token_B)
        self._pairs.pop(key)

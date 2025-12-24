"""
Test suite for tokenizer inference functionality
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tokenizer.tokenizer import Tokenizer


def test_common_words_not_split():
    """Test that common words exist as single tokens and aren't split."""
    checkpoint_path = Path(__file__).parent.parent.parent / 'tokenizer/trained/owt_train/final_0032000_inference.pkl'
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    tokenizer = Tokenizer.load_for_inference(checkpoint_path)
    
    # Test cases: (text, expected_token_count)
    test_cases = [
        (" is", 1),           # Should be single token #321
        ("hello", 1),         # Should be single token #31898
        (" the", 1),          # Should be single token #262
        ("This is a test", 4), # ['This', ' is', ' a', ' test']
    ]
    
    print("Testing tokenization...")
    all_passed = True
    
    for text, expected_count in test_cases:
        tokens = tokenizer.inference_on_text(text, ret_type='tokens')
        total_tokens = sum(len(token_list) for token_list in tokens)
        
        if total_tokens == expected_count:
            print(f"✓ '{text}' -> {total_tokens} tokens")
        else:
            print(f"✗ '{text}' -> {total_tokens} tokens (expected {expected_count})")
            all_passed = False
            # Show what we got
            for token_list in tokens:
                for token in token_list:
                    decoded = token.byte_arr.decode('utf-8', errors='replace')
                    print(f"    [{token.token_idx}] '{decoded}'")
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    return all_passed


def test_byte_concatenation_merging():
    """Test that byte concatenation merging works correctly."""
    checkpoint_path = Path(__file__).parent.parent.parent / 'tokenizer/trained/owt_train/final_0032000_inference.pkl'
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    tokenizer = Tokenizer.load_for_inference(checkpoint_path)
    
    # Verify that " is" merges correctly via byte concatenation
    # Start with [' ', 'i', 's'] -> should become [' is']
    text = " is"
    tokens = tokenizer.inference_on_text(text, ret_type='tokens')
    
    assert len(tokens) == 1, f"Expected 1 pretoken, got {len(tokens)}"
    assert len(tokens[0]) == 1, f"Expected 1 token, got {len(tokens[0])}"
    
    token = tokens[0][0]
    assert token.byte_arr == b' is', f"Expected b' is', got {token.byte_arr}"
    assert token.token_idx == 321, f"Expected token #321, got #{token.token_idx}"
    
    print("✓ Byte concatenation merging works correctly")
    return True


if __name__ == "__main__":
    print("="*80)
    print("TOKENIZER INFERENCE TESTS")
    print("="*80)
    
    test_common_words_not_split()
    print()
    test_byte_concatenation_merging()

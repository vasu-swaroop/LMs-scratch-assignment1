"""
Tokenizer Inference Examples

This script demonstrates how to use the trained tokenizer for text encoding.
Copy these code blocks into your tokenizer_inference.ipynb notebook.

Note: Make sure you have the 'regex' package installed: pip install regex
"""

from tokenizer import Tokenizer

# =============================================================================
# CELL 1: Load the tokenizer
# =============================================================================
checkpoint_path = '/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl'
tokenizer = Tokenizer.load_for_inference(checkpoint_path)
print(f"Loaded tokenizer with vocab size: {tokenizer.token_registery.num_tokens}")


# =============================================================================
# CELL 2: Basic tokenization - get token IDs
# =============================================================================
text = "Hello world! This is a test of the BPE tokenizer."

# Get token indices (most common use case)
token_ids = tokenizer.inference_on_text(text, ret_type='int')
print(f"Input text: {text}")
print(f"\nToken IDs: {token_ids}")
print(f"Total tokens: {sum(len(ids) for ids in token_ids)}")


# =============================================================================
# CELL 3: Detailed token inspection
# =============================================================================
# Get Token objects for detailed inspection
tokens = tokenizer.inference_on_text(text, ret_type='tokens')

print("Detailed token breakdown:")
for i, token_list in enumerate(tokens):
    print(f"\nPretoken {i}:")
    for token in token_list:
        try:
            decoded = token.byte_arr.decode('utf-8')
        except:
            decoded = str(token.byte_arr)
        print(f"  ID {token.token_idx:5d}: '{decoded}' (freq: {token.token_freq})")


# =============================================================================
# CELL 4: Decode tokens back to text
# =============================================================================
def decode_token_ids(tokenizer, token_ids_nested):
    """Decode token IDs back to text."""
    decoded_bytes = b''.join([
        tokenizer.token_registery.get_token(token_id).byte_arr 
        for token_list in token_ids_nested 
        for token_id in token_list
    ])
    return decoded_bytes.decode('utf-8', errors='replace')

# Test decoding
decoded_text = decode_token_ids(tokenizer, token_ids)
print(f"Original:  {text}")
print(f"Decoded:   {decoded_text}")
print(f"Match: {text == decoded_text}")


# =============================================================================
# CELL 5: Calculate compression ratio
# =============================================================================
byte_count = len(text.encode('utf-8'))
token_count = sum(len(ids) for ids in token_ids)
compression_ratio = byte_count / token_count

print(f"Bytes: {byte_count}")
print(f"Tokens: {token_count}")
print(f"Compression ratio: {compression_ratio:.2f}x")
print(f"(Each token represents ~{compression_ratio:.2f} bytes on average)")


# =============================================================================
# CELL 6: Try with custom text
# =============================================================================
custom_text = "The quick brown fox jumps over the lazy dog."

custom_token_ids = tokenizer.inference_on_text(custom_text, ret_type='int')
print(f"Text: {custom_text}")
print(f"Token IDs: {custom_token_ids}")
print(f"Token count: {sum(len(ids) for ids in custom_token_ids)}")

# Decode back
decoded = decode_token_ids(tokenizer, custom_token_ids)
print(f"Decoded: {decoded}")


# =============================================================================
# CELL 7: Batch processing example
# =============================================================================
texts = [
    "Machine learning is amazing!",
    "Natural language processing with BPE.",
    "Tokenization is the first step."
]

print("Batch tokenization:")
for i, text in enumerate(texts):
    ids = tokenizer.inference_on_text(text, ret_type='int')
    total_tokens = sum(len(id_list) for id_list in ids)
    print(f"{i+1}. \"{text}\"")
    print(f"   Tokens: {total_tokens}, IDs: {ids}\n")

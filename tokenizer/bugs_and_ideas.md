
## Bugs Faced

- [x] Did not delete the token pair from the token pair registry when I found the most frequent.
- [x] After merging, I did not apply a recursive merge update on the token_pair.
- [x] Was not updating the token list when creating new.
- [x] Double counting of `A A A A`. (Used LLM to find and debug).
- [x] Proper merging during word tokenization inference. I am just checking the existence in the tokenizer

> Adding asserts really helped with testing.

## Ideas

### 1. Context-Specific Tokenization patterns
We should use different tokenization patterns for different texts.

- **Text information**: BPE makes sense.
- **Coding information**: We should use a separate tokenizer.
- **Maths**: We should come up with a separate tokenizer.

### 2. Adaptation
Add `|<beginofmaths>|` `|<endofmaths>|` and then add the tokenization procedure where we can add some maths tokenization, which is more than a lookup, and is a function.

#### Examples:
- `1.6789` should be tokenized as: `pre_dec` `<decimal>` `post_dec`
- `1985-593` should be tokenized as:
  ```
  |<beginofmaths>| <beginofnum> <val_A> <endofnum> <minus> <beginofnum> <val_B> <endofnum> |<endofmaths>|
  ```
  Instead of: `<1><9><8><5> <val_A>=LSTM[<1>,<9>,<8>,<5>,]`

#### Challenges:
- Statistics.
- Separating different numbers (Dates, Ranges, etc.).

> We can use an LLM as a synthetic augmentor.

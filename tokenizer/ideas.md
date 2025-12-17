1. We should use different tokenization patterns for different texts
1.1 For text information, BPE makes sense
1.2 For coding information, we should use a separate tokenizer
1.3 For maths also, we should come up with a separte tokenizer

Adaptation:
    Add |<beginofmaths>| |<endofmaths>| and then add the tokenization procedure where we can add some maths tokenization, which is more than a lookup, and is a function. 

    so 1.6789, should be tokenized as pre_dec <decimal> post_dec
    1985-593 should be tokenized as |<beginofmaths>| <beginofnum> <val_A> <endofnum> <minus> <beginofnum> <val_B> <endofnum>   |<endofmaths>| instead of <1><9><8><5> <val_A>=LSTM[<1>,<9>,<8>,<5>,]

    The problem would be statistics

    And for separting the different numbers
    1. Dates
    2. Ranges
    3. etc.

    we can use an LLM as a synthetic augmentor
    
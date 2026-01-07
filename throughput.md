With this stack, we train the 7B model
at 7700 tokens per second per GPU and the 32B model at 1960 tokens per second per GPU at a sequence
length of 8192, using bfloat16 precision throughout.  This corresponds to roughly 43% and 41% MFU,
respectively. 
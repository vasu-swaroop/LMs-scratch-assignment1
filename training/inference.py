from tokenizer.tokenizer import Tokenizer

test_text=open('/data3/vasu/projects/LMs-scratch-assignment1/data/overfiting_100_lines.txt','r').read()
tokenizer=Tokenizer.load_for_inference('/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl')

tokens=tokenizer.inference_on_text(test_text)
#TODO: Increase endoftext frequency aptly

print(tokens[-1])
token_list=[token.byte_arr.decode() for word in tokens for token in word]
# print('|'.join(token_list))
detokinze=''.join(token_list)
byte_stream_len=len(test_text.encode())
num_tokens=len([token for word in tokens for token in word])

print(byte_stream_len, num_tokens)
print("approx compression ratio:",byte_stream_len/num_tokens)
assert detokinze==test_text, "Error on tokenization"





# if __name__=='__main__':
    
#     # moe_mla_config = MLA_config(
#     #     latent_dim_q=8,
#     #     latent_dim_kv=16,
#     #     dim_content=512,
#     #     dim_pos=128,
#     #     num_heads=8,
#     # )
    
#     # moe_ffn_config = MOE_FFN_config(
#     #     num_shared_experts=2,      # Shared experts (general number = 2)
#     #     num_routing_experts=3,      # Total routing experts available
#     #     num_selected_experts=2,     # Top-k selection (selects top 2 out of 6)
#     #     expert_dim=512,           # Hidden dimension for expert FFNs
#     #     activation=Activation.RELU,
#     #     router_type=RouterType.LEARNED
#     # )
    
#     # moe_model_config = ModelConfig(
#     #     mla_config=moe_mla_config,
#     #     moe_ffn_config=moe_ffn_config,
#     #     model_dim=512,
#     #     transformer_depth=10,
#     #     vocab_length=32_000
#     # )
    
#     # moe_train_settings = TrainSettings(
#     #     optimizer='adam',
#     #     lr=1e-5,                    # Lower learning rate for MOE models
#     #     num_epochs=20,
#     #     batch_size=8,              # Reduced for stability/OOM
#     #     cur_steps=0,
#     #     cur_epoch=0,
#     #     resume_ckpt=True,
#     #     train_steps=1000000,
#     #     seq_len=1024,
#     #     prng_key=jax.random.PRNGKey(42),
#     #     data_path=Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/overfiting_test_tineystoruies_train'),
#     #     val_data_path=Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/overfiting_test_tineystoruies_valid'),
#     #     grad_accumulation=4,
#     #     num_gpus=2,
#     #     use_wandb=True,
#     #     save_every=10000
#     # )

#     # train_dataset=Data(moe_train_settings.data_path,moe_train_settings.batch_size,moe_train_settings.train_steps, moe_train_settings.seq_len)
#     # val_dataset=Data(moe_train_settings.val_data_path,moe_train_settings.batch_size,moe_train_settings.train_steps, moe_train_settings.seq_len)
#     # trainer = Training(moe_train_settings, moe_model_config)
#     # trainer.train(train_dataset, val_dataset)

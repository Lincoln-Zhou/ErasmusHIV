import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="medgemma-27b-text-it.i1-Q4_K_M.gguf",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=32768,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
_ = model.generate(streamer=text_streamer, max_new_tokens=64)

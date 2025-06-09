import torch
from huggingface_hub import login

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


login(token='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')


def run_unsloth(prompt: str, pipe):
    # Execute a single prompt and get parsed response from LLM
    model, tokenizer = pipe

    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    )

    gen_tokens = outputs[0][inputs["input_ids"].size(-1):]

    response = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True,
    )

    return response


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        max_seq_length=32768,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma3",
    )

    run_unsloth(prompt="What are you doing?", pipe=(model, tokenizer))


if __name__ == '__main__':
    main()

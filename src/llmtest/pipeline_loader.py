from langchain import (
    HuggingFaceHub,
    HuggingFacePipeline
)

from transformers import pipeline


def getPipeLIneWithDeviceMap(
        model,
        tokenizer,
        task="text-generation",
        use_cache=True,
        device_map="auto",
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        max_new_tokens=256,
):
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_cache=use_cache,
        device_map=device_map,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens
    )


def getPipeLIneWithoutDeviceMap(
        model,
        tokenizer,
        task="text-generation",
        use_cache=True,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        max_new_tokens=256,
):
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_cache=use_cache,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens
    )

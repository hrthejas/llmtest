from langchain import (
    HuggingFaceHub,
    HuggingFacePipeline
)
import torch
from llmtest import model_loader
from transformers import pipeline


def load_simple_pipeline(model_id, task, max_new_tokens, device_map):
    pipe = pipeline(
        model=model_id,
        task=task,
        device_map=device_map,
        torch_dtype=torch.float16,
        max_new_tokens=max_new_tokens
    )

    return HuggingFacePipeline(pipeline=pipe)

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

def load_pipeline(device_map, do_sample, max_new_tokens, model_class, model_id, num_return_sequences, set_device_map,
                  task, tokenizer_class, top_k, use_4bit_quantization, use_cache):
    model = model_loader.load_model(model_id, use_4bit_quantization, model_class, device_map=device_map,
                                    set_device_map=set_device_map)
    tokenizer = model_loader.getTokenizer(model_id, tokenizer_class)
    if set_device_map:
        pipe = getPipeLIneWithDeviceMap(
            model,
            tokenizer,
            task=task,
            use_cache=use_cache,
            device_map=device_map,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        pipe = getPipeLIneWithoutDeviceMap(
            model,
            tokenizer,
            task=task,
            use_cache=use_cache,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )
        return HuggingFacePipeline(pipeline=pipe)
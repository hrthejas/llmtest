import huggingface_hub
import torch
from llmtest import pipeline_loader, model_loader
from getpass import getpass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


def hf_login():
    from huggingface_hub import login
    token = getpass("Paste your HF API key here and hit enter:")
    login(token=token)


def save_model(model, tokenizer, out_model_name, max_shard_size, safe_serialization):
    model.push_to_hub(out_model_name, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
    tokenizer.push_to_hub(save_model)


def shard_model(
        model_id,
        out_model_name,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        device_map="auto",
        set_device_map=False,
        torch_dtype=torch.bfloat16,
        offload_folder='offload',
        max_shard_size="3GB",
        safe_serialization=True
):
    hf_login()
    model = model_loader.getModelForSharding(model_id, model_class, device_map=device_map, torch_dtype=torch_dtype,
                                             offload_folder=offload_folder)
    tokenizer = model_loader.getTokenizer(model_id, tokenizer_class)
    save_model(model, tokenizer, out_model_name, max_shard_size, safe_serialization)

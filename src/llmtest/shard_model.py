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
    tokenizer.push_to_hub(out_model_name)


def shard_model(
        model_id,
        out_model_name,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        device_map="auto",
        set_device_map=True,
        torch_dtype=torch.bfloat16,
        offload_folder='offload',
        max_shard_size="3GB",
        safe_serialization=True
):
    hf_login()
    additional_model_args = {}
    additional_model_args['offload_folder'] = offload_folder
    custom_quantinzation_conf = None
    use_quantization = False
    is_gptq = False
    is_gglm = False
    use_triton = False
    additional_tokenizer_args = None
    model = model_loader.get_model(model_id, model_class, device_map, use_quantization, additional_model_args, is_gptq,
                                   is_gglm, custom_quantinzation_conf,
                                   safe_serialization, use_triton, set_device_map)
    tokenizer = model_loader.get_tokenizer(model_id, tokenizer_class, additional_tokenizer_args)
    save_model(model, tokenizer, out_model_name, max_shard_size, safe_serialization)

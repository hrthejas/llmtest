import torch
from transformers import BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    GenerationConfig,
)


def getTokenizer(model_id, tokenizer_class=AutoTokenizer):
    tokenizer = tokenizer_class.from_pretrained(model_id)
    return tokenizer;


def getQuantizedModel(model_id, model_class=AutoModelForCausalLM, device_map="auto", set_device_map=False):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = model_class.from_pretrained(model_id, device_map=device_map, quantization_config=quantization_config,
                                        trust_remote_code=True)
    return model


def getModelForSharding(model_id, model_class=AutoModelForCausalLM, torch_dtype=torch.bfloat16, device_map='auto',
                        offload_folder='offload'):
    model = model_class.from_pretrained(model_id, device_map=device_map, trust_remote_code=True,
                                        torch_dtype=torch_dtype, offload_folder=offload_folder)
    return model


def getNonQuantizedModel(model_id, model_class=AutoModelForCausalLM, device_map="auto", set_device_map=False):
    if set_device_map == True:
        model = model_class.from_pretrained(model_id, trust_remote_code=True, device_map=device_map)
        return model
    else:
        model = model_class.from_pretrained(model_id, trust_remote_code=True)
        return model


def loadModel(model_id, use_4bit_quantization=False, model_class=AutoModelForCausalLM, device_map="auto",
              set_device_map=False):
    if use_4bit_quantization == True:
        return getQuantizedModel(model_id, model_class, device_map="auto", set_device_map=False)
    else:
        return getNonQuantizedModel(model_id, model_class, device_map="auto", set_device_map=False)

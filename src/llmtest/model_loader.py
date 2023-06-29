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


def getQuantizedModel(model_id, model_class=AutoModelForCausalLM):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = model_class.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config,
                                        trust_remote_code=True)
    return model


def getNonQuantizedModel(model_id, model_class=AutoModelForCausalLM):
    model = model_class.from_pretrained(model_id, trust_remote_code=True)
    return model


def loadModel(model_id, use_4bit_quantization=False, model_class=AutoModelForCausalLM):
    if use_4bit_quantization == True:
        return getQuantizedModel(model_id, model_class)
    else:
        return getNonQuantizedModel(model_id, model_class)

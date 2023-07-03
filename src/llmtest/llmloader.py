from langchain import (
    HuggingFacePipeline
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from llmtest import pipeline_loader, model_loader


def getLLM(
        model_id,
        use_4bit_quantization=False,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        task="text-generation",
        use_cache=True,
        device_map="auto",
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        max_new_tokens=256,
        set_device_map=False,
        use_simple_llm_loader=False
):
    if use_simple_llm_loader:
        return pipeline_loader.load_simple_pipeline(model_id, task, max_new_tokens, device_map)
    else:
        return pipeline_loader.load_pipeline(device_map, do_sample, max_new_tokens, model_class, model_id,
                                             num_return_sequences,
                                             set_device_map, task, tokenizer_class, top_k, use_4bit_quantization,
                                             use_cache)

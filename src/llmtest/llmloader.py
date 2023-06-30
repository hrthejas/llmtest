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
        set_device_map=False
):
    model = model_loader.loadModel(model_id, use_4bit_quantization, model_class, device_map=device_map,
                                   set_device_map=set_device_map)
    tokenizer = model_loader.getTokenizer(model_id, tokenizer_class)
    if set_device_map:
        pipeline = pipeline_loader.getPipeLIneWithDeviceMap(
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
        return HuggingFacePipeline(pipeline=pipeline)
    else:
        pipeline = pipeline_loader.getPipeLIneWithoutDeviceMap(
            model,
            tokenizer,
            task=task,
            use_cache=use_cache,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )
        return HuggingFacePipeline(pipeline=pipeline)

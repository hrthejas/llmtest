import torch
from langchain import (
    HuggingFacePipeline
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    GenerationConfig,
)

from llmtest import pipeline_loader, model_loader, constants


def get_llm(model_id,
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            task="text-generation",
            use_quantization=False,
            max_new_tokens=296,
            is_gptq_model=False,
            is_gglm_model=False,
            device_map="auto",
            use_safetensors=False,
            use_triton=False,
            additional_model_args=None,
            additional_pipeline_args=None,
            additional_tokenizer_args=None,
            custom_quantization_config=None, pass_device_map=False):
    tokenizer = model_loader.get_tokenizer(model_id, tokenizer_class, additional_tokenizer_args)
    model = model_loader.get_model(model_id, model_class, device_map, use_quantization, additional_model_args,
                                   is_gptq_model, is_gglm_model, custom_quantization_config, use_safetensors,
                                   use_triton, pass_device_map)
    pipeline = pipeline_loader.get_pipeline(model, task, tokenizer, max_new_tokens, additional_pipeline_args)
    return HuggingFacePipeline(pipeline=pipeline)


def load_llm(
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
        use_simple_llm_loader=False,
        is_quantized_gptq_model=False,
        use_safetensors=False,
        use_triton=False,
        custom_quantiztion_config=None,
        additional_model_Args=None,
        is_gglm_model=False,
        additional_pipeline_args=None,
        additional_tokenizer_args=None,
        set_eos_token=True,
        set_pad_token=True,
):
    if additional_pipeline_args is None:
        additional_pipeline_args = {}

    pipe = None
    if use_simple_llm_loader:
        pipe = pipeline_loader.get_pipeline_from_model_id(model_id, task, max_new_tokens, additional_model_Args,
                                                          device_map,
                                                          torch_dtype=torch.bfloat16)
    else:
        tokenizer = model_loader.get_tokenizer(model_id, tokenizer_class, additional_tokenizer_args)
        model = model_loader.get_model(model_id, model_class, device_map, use_4bit_quantization, additional_model_Args,
                                       is_quantized_gptq_model, is_gglm_model, custom_quantiztion_config,
                                       use_safetensors,
                                       use_triton, set_device_map)

        additional_pipeline_args['top_k'] = top_k
        additional_pipeline_args['num_return_sequences'] = num_return_sequences

        if use_cache:
            additional_pipeline_args['use_cache'] = use_cache
        if do_sample:
            additional_pipeline_args['do_sample'] = do_sample
        if set_eos_token:
            additional_pipeline_args['eos_token_id'] = tokenizer.eos_token_id
        if set_pad_token:
            additional_pipeline_args['pad_token_id'] = tokenizer.eos_token_id

        print("Additional pipeline args ")
        print(additional_pipeline_args)

        if set_device_map:
            print("Creating a pipeline with device map")

            pipe = pipeline_loader.get_pipeline(model=model, task=task, tokenizer=tokenizer,
                                                max_new_tokens=max_new_tokens,
                                                additional_pipeline_args=additional_pipeline_args, use_fast=True,
                                                device_map=device_map)
        else:
            pipe = pipeline_loader.get_pipeline(model=model, task=task, tokenizer=tokenizer,
                                                max_new_tokens=max_new_tokens,
                                                additional_pipeline_args=additional_pipeline_args, use_fast=True,
                                                device_map=None)

    return HuggingFacePipeline(pipeline=pipe)

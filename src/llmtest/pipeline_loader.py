from transformers import pipeline


def get_pipeline(model, task, tokenizer, max_new_tokens, additional_pipeline_args, use_fast=True, device_map=None,
                 torch_dtype=None):
    if additional_pipeline_args is None:
        additional_pipeline_args = {}
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        use_fast=use_fast,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **additional_pipeline_args
    )


def get_pipeline_test_props(model, task, tokenizer, max_new_tokens, device_map=None):
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
        top_k=1,
        num_return_sequences=1,
        use_cache=True,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )


def get_pipeline_from_model_id(model_id, task, max_new_tokens, additional_pipeline_args, device_map=None,
                               torch_dtype=None):
    if additional_pipeline_args is None:
        additional_pipeline_args = {}
    return pipeline(
        model=model_id,
        task=task,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        **additional_pipeline_args
    )

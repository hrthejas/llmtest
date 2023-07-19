import torch
from transformers import BitsAndBytesConfig


def get_gptq_model(model_id, device_map, use_quantization, use_safetensors, use_triton, custom_quantization_conf,
                   additional_model_args, model_basename=None):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    if additional_model_args is None:
        additional_model_args = {}

    if use_quantization:
        if custom_quantization_conf is None:
            raise Exception(
                "quantization configurations are not passed pass parameter 'custom_quantization_conf' "
                "using BaseQuantizeConfig")

    return AutoGPTQForCausalLM.from_quantized(model_id, trust_remote_code=True, device=device_map,
                                              use_safetensors=use_safetensors,
                                              use_triton=use_triton, quantize_config=custom_quantization_conf,
                                              **additional_model_args)


def get_generic_model(model_id, model_class, device_map, use_quantization, custom_quantization_conf,
                      additional_model_args, pass_device_map, set_torch_dtype, torch_dtype):
    if additional_model_args is None:
        additional_model_args = {}

    if use_quantization:
        if custom_quantization_conf is None:
            print(
                "Using default quntization config from bits and bytes use 'custom_quantization_conf' if you want ot "
                "change")
            custom_quantization_conf = get_quantization_config()

    if use_quantization:
        print("Loading model " + model_id + " with quantization")
        return model_class.from_pretrained(model_id, device_map=device_map,
                                           quantization_config=custom_quantization_conf,
                                           trust_remote_code=True, **additional_model_args)
    else:
        print("Loading model " + model_id)
        if set_torch_dtype:
            if additional_model_args is None:
                additional_model_args = {}
            additional_model_args['torch_dtype'] = torch_dtype

        if additional_model_args is not None:
            print("Setting additional model args")
            print(additional_model_args)

        if pass_device_map:
            return model_class.from_pretrained(model_id, trust_remote_code=True, device_map=device_map,
                                               **additional_model_args)
        else:
            return model_class.from_pretrained(model_id, trust_remote_code=True, **additional_model_args)


def get_model(model_id, model_class, device_map, use_quantization, additional_model_args, is_gptq_model, is_gglm_model,
              custom_quantization_conf, use_safetensors, use_triton, pass_device_map, set_dorch_dtype, torch_dtype,
              model_basename):
    if is_gptq_model:
        return get_gptq_model(model_id, device_map, use_quantization, use_safetensors, use_triton,
                              custom_quantization_conf, additional_model_args, model_basename)
    elif is_gglm_model:
        return None
    else:
        return get_generic_model(model_id, model_class, device_map, use_quantization, custom_quantization_conf,
                                 additional_model_args, pass_device_map, set_dorch_dtype, torch_dtype)


def get_tokenizer(model_id, tokenizer_class, additional_tokenizer_args):
    if additional_tokenizer_args is not None:
        return tokenizer_class.from_pretrained(model_id, **additional_tokenizer_args)
    else:
        return tokenizer_class.from_pretrained(model_id)


def get_quantization_config():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

import torch
args = {"model_id":"/content/drive/Shareddrives/Engineering/Chatbot/Models/thr-wlm-15b-3gb",
    "use_4bit_quantization":False,"set_device_map":True,"max_new_tokens":512,
    "device_map":"auto",
    "use_simple_llm_loader":True,
    "is_quantized_gptq_model":False,
    "set_torch_dtype":False,
    "torch_dtype":torch.bfloat16
}

from llmtest.ChatApp import ChatApp
chat_app = ChatApp()
chat_app.__int__(**args)
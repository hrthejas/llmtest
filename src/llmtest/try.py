import torch

from llmtest.IWXBot import IWXBot

args = {
    "mount_gdrive": False,
    "index_base_path": "/Users/thejas/Downloads/chatbot/indexes/hf/",
    "use_4bit_quantization": False,
    "max_new_tokens": 800,
    "use_simple_llm_loader": False,
}

chatbot = IWXBot(**args)
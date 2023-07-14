import torch

from llmtest import constants
from llmtest.IWXGPT import IWXGPT
from llmtest.IWXRetriever import IWXRetriever

args = {
    "index_base_path": "/Users/thejas/Downloads/chatbot/indexes/openai/",
    "max_new_tokens": 800,
    "mount_gdrive": False
}

chatbot = IWXGPT(**args)
embeddings = chatbot.get_embeddings()
chatbot.initialize_chat()
chatbot.start_chat(api_prompt_template=constants.DEFAULT_PROMPT_FOR_API_2)

retr = IWXRetriever()

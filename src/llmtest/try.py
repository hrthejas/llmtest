import torch

from llmtest import constants
from llmtest.IWXGPT import IWXGPT

args = {
    "mount_gdrive": False,
    "index_base_path": "/Users/thejas/Downloads/chatbot/indexes/openai/",
    "max_new_tokens": 800,
}

chatbot = IWXGPT(**args)
embeddings = chatbot.get_embeddings()
chatbot.overwrite_vector_store(docs_type="CSV", index_base_path="/Users/thejas/Downloads/chatbot/indexes/openai/",
                               index_name="api_index_csv", chunk_size=4000, chunk_overlap=20, embeddings=embeddings,
                               docs_base_path="/Users/thejas/Downloads/chatbot/restapicsv")

chatbot.initialize_chat()
chatbot.start_chat(api_prompt_template=constants.DEFAULT_PROMPT_FOR_API_2)
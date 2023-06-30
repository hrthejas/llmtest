from llmtest import contants, startchat


def start(load_gpt_model=True, load_local_model=True, local_model_id=contants.DEFAULT_MODEL_NAME,
          docs_base_path=contants.DOCS_BASE_PATH, index_base_path=contants.INDEX_BASE_PATH,
          index_name_prefix=contants.INDEX_NAME_PREFIX,
          max_new_tokens=contants.MAX_NEW_TOKENS, use_4bit_quantization=contants.USE_4_BIT_QUANTIZATION,
          use_prompt=True, prompt=contants.QUESTION_PROMPT, set_device_map=contants.SET_DEVICE_MAP):
    import gradio as gr

    retriever = startchat.get_embedding_retriever(index_base_path=index_base_path, index_name_prefix=index_name_prefix,
                                                  docs_base_path=docs_base_path)
    if load_gpt_model:
        openai_llm = startchat.load_openai_model(retriever)
    else:
        openai_llm = None

    if load_local_model:
        local_llm = startchat.load_local_model(retriever, model_id=local_model_id,
                                               use_4bit_quantization=use_4bit_quantization,
                                               max_new_tokens=max_new_tokens, set_device_map=set_device_map)
    else:
        local_llm = None

    def chatbot(message):
        if use_prompt:
            final_question = prompt + '\n' + message
        else:
            final_question = message
        if openai_llm is not None:
            response1 = startchat.get_chat_gpt_result(openai_llm, final_question)['result']
        else:
            response1 = "Seams like open ai model is not loaded or not requested to give answer"
        if local_llm is not None:
            response2 = startchat.get_local_model_result(local_llm, final_question)['result']
        else:
            response2 = "Seams like iwxchat model is not loaded or not requested to give answer"

        return response1, response2

    iface = gr.Interface(fn=chatbot, inputs=gr.inputs.Textbox(label="Enter your question"),
                         outputs=[gr.outputs.Textbox(label="Response from chatgpt"),
                                  gr.outputs.Textbox(label="Response from iwx")])
    iface.launch()

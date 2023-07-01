import gradio as gr
import time
import threading
from llmtest import contants, startchat, ingest, storage


def get_question(user_input, use_prompt, prompt):
    if use_prompt:
        return prompt + '\n' + user_input
    else:
        return user_input


def record_answers(query, open_ai_answer, local_model_answer):
    try:
        storage.insert_data(contants.USER_NAME, query, open_ai_answer, local_model_answer)
    except:
        print("Error while recording answers to db")
        pass


def start(load_gpt_model=True, load_local_model=True, local_model_id=contants.DEFAULT_MODEL_NAME,
          docs_base_path=contants.DOCS_BASE_PATH, index_base_path=contants.INDEX_BASE_PATH,
          index_name_prefix=contants.INDEX_NAME_PREFIX,
          max_new_tokens=contants.MAX_NEW_TOKENS, use_4bit_quantization=contants.USE_4_BIT_QUANTIZATION,
          use_prompt=True, prompt=contants.QUESTION_PROMPT, set_device_map=contants.SET_DEVICE_MAP, mount_gdrive=True,
          share_chat_ui=True, debug=False, gdrive_mount_base_bath=contants.GDRIVE_MOUNT_BASE_PATH,
          openai_llm=None, local_llm=None, hf_retriever=None, oi_retriever=None, load_retriever=True,
          device_map=contants.DEFAULT_DEVICE_MAP):
    if mount_gdrive:
        ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)

    if load_retriever and hf_retriever is None:
        hf_retriever = startchat.get_embedding_retriever(index_base_path=index_base_path,
                                                         index_name_prefix=index_name_prefix,
                                                         docs_base_path=docs_base_path)
    if load_retriever and oi_retriever is None:
        oi_retriever = startchat.get_embedding_retriever_openai(index_base_path=index_base_path,
                                                                index_name_prefix=index_name_prefix,
                                                                docs_base_path=docs_base_path)

    if load_gpt_model and openai_llm is None:
        openai_llm = startchat.load_openai_model(oi_retriever)
    else:
        openai_llm = None

    if load_local_model and local_llm is None:
        local_llm = startchat.load_local_model(hf_retriever, model_id=local_model_id,
                                               use_4bit_quantization=use_4bit_quantization,
                                               max_new_tokens=max_new_tokens, set_device_map=set_device_map)
    else:
        local_llm = None

    with gr.Blocks(theme="gradio/monochrome", mode="IWX CHATBOT", title="IWX CHATBOT") as demo:
        chatbot_1 = gr.Chatbot(label="OPEN AI")
        chatbot_2 = gr.Chatbot(label="IWX AI")
        msg = gr.Textbox(label="User Question")
        clear = gr.ClearButton([msg, chatbot_1, chatbot_2])

        def user(user_message, history_1, history_2):
            return gr.update(value="", interactive=False), history_1 + [[user_message, None]], history_2 + [
                [user_message, None]]

        def bot_1(history_1):
            query = get_question(history_1[-1][0], use_prompt, prompt)
            if openai_llm is not None:
                bot_message = startchat.get_chat_gpt_result(openai_llm, query)['result']
            else:
                bot_message = "Seams like open ai model is not loaded or not requested to give answer"
            history_1[-1][1] = ""
            for character in bot_message:
                history_1[-1][1] += character
                time.sleep(0.0005)
                yield history_1

        def bot_2(history_1, history_2):
            query = get_question(history_2[-1][0], use_prompt, prompt)
            if local_llm is not None:
                bot_message = startchat.get_local_model_result(local_llm, query)['result']
            else:
                bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
            record_answers(history_2[-1][0], history_1[-1][1], bot_message)
            history_2[-1][1] = ""
            for character in bot_message:
                history_2[-1][1] += character
                time.sleep(0.0005)
                yield history_2

        response = msg.submit(user, [msg, chatbot_1, chatbot_2], [msg, chatbot_1, chatbot_2], queue=False).then(
            bot_1, chatbot_1, chatbot_1
        )
        response.then(bot_2, [chatbot_1, chatbot_2], chatbot_2)
        response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

        demo.queue()
        demo.launch(share=share_chat_ui, debug=debug)

    # with gr.Blocks(theme="gradio/monochrome", mode="IWX CHATBOT", title="IWX CHATBOT") as demo:
    #     chatbot_1 = gr.Chatbot(label="OPEN AI")
    #     chatbot_2 = gr.Chatbot(label="IWX AI")
    #     msg = gr.Textbox(label="User Question")
    #     clear = gr.ClearButton([msg, chatbot_1, chatbot_2])
    #
    #     def user(user_message, history_1, history_2):
    #         return gr.update(value="", interactive=False), history_1 + [[user_message, None]], history_2 + [
    #             [user_message, None]]
    #
    #     def bot_1(history_1):
    #         query = get_question(history_1[-1][0], use_prompt, prompt)
    #         if openai_llm is not None:
    #             bot_message = startchat.get_chat_gpt_result(openai_llm, query)['result']
    #         else:
    #             bot_message = "Seams like open ai model is not loaded or not requested to give answer"
    #         history_1[-1][1] = ""
    #         for character in bot_message:
    #             history_1[-1][1] += character
    #             time.sleep(0.0005)
    #             yield history_1
    #
    #     def bot_2(history_2):
    #         query = get_question(history_2[-1][0], use_prompt, prompt)
    #         if local_llm is not None:
    #             bot_message = startchat.get_local_model_result(local_llm, query)['result']
    #         else:
    #             bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
    #         history_2[-1][1] = ""
    #         for character in bot_message:
    #             history_2[-1][1] += character
    #             time.sleep(0.0005)
    #             yield history_2
    #
    #     response = msg.submit(user, [msg, chatbot_1, chatbot_2], [msg, chatbot_1, chatbot_2], queue=False)
    #
    #     thread_1 = threading.Thread(target=response.then, args=(bot_1, chatbot_1, chatbot_1))
    #     thread_1.start()
    #
    #     thread_2 = threading.Thread(target=response.then, args=(bot_2, chatbot_2, chatbot_2))
    #     thread_2.start()
    #
    #     response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
    #
    #     demo.queue()
    #     demo.launch(share=share_chat_ui, debug=debug)

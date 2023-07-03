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

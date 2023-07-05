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


# def get_retriever_for_chain(docs_base_path, index_base_path, index_name_prefix,
#                             embeddings,
#                             index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
#                             read_md_docs=True,
#                             chunk_size=1000, chunk_overlap=100, search_type="similarity", search_kwargs={"k": 1}):
#
#     vector_store = get_vector_store(docs_base_path, index_base_path, index_name_prefix, embeddings,
#                                     index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
#                                     read_md_docs=True,
#                                     chunk_size=1000, chunk_overlap=100)
#
#     if vector_store is not None:
#         return get_retriever_from_store(vector_store, search_type=search_type, search_kwargs=search_kwargs)
#     else:
#         raise Exception("Sorry, Unknown index_type : ")
#
#
# def get_retriever_for_openai_chain(docs_base_path, index_base_path, index_name_prefix,
#                                    index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
#                                    read_md_docs=True,
#                                    chunk_size=1000, chunk_overlap=100, search_type="similarity",
#                                    search_kwargs={"k": 4}):
#     embeddings = OpenAIEmbeddings()
#     vector_store = get_vector_store(docs_base_path, index_base_path, index_name_prefix, embeddings,
#                                     index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
#                                     read_md_docs=True,
#                                     chunk_size=1000, chunk_overlap=100)
#
#     if vector_store is not None:
#         return get_retriever_from_store(vector_store, search_type=search_type, search_kwargs=search_kwargs)
#     else:
#         raise Exception("Sorry, Unknown index_type : ")

# def get_embedding_retriever(docs_base_path=constants.DOCS_BASE_PATH,
#                             index_base_path=constants.INDEX_BASE_PATH,
#                             index_name_prefix=constants.INDEX_NAME_PREFIX):
#     index_base_path = index_base_path + "/hf/"
#     return vectorstore.get_retriever_for_chain(docs_base_path=docs_base_path,
#                                                index_base_path=index_base_path,
#                                                index_name_prefix=index_name_prefix)
#
#
# def get_embedding_retriever_openai(docs_base_path=constants.DOCS_BASE_PATH,
#                                    index_base_path=constants.INDEX_BASE_PATH,
#                                    index_name_prefix=constants.INDEX_NAME_PREFIX):
#     os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
#     index_base_path = index_base_path + "/openai/"
#     return vectorstore.get_retriever_for_openai_chain(docs_base_path=docs_base_path,
#                                                       index_base_path=index_base_path,
#                                                       index_name_prefix=index_name_prefix)

# pipe_kwargs_names = [
#     "use_cache",
#     "do_sample",
#     "top_k",
#     "num_return_sequences",
#     "proxies",
#     "eos_token_id",
#     "pad_token_id",
# ]
# pipe_kwargs = {name: pipeline_args.pop(name) for name in pipe_kwargs_names if name in pipeline_args}
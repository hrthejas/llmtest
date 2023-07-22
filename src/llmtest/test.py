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

# def start(load_gpt_model=False, load_local_model=True, local_model_id=constants.DEFAULT_MODEL_NAME,
#           docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
#           docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX, api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
#           max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
#           use_prompt=False, prompt=constants.API_QUESTION_PROMPT, set_device_map=constants.SET_DEVICE_MAP,
#           mount_gdrive=True,
#           share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
#           device_map=constants.DEFAULT_DEVICE_MAP, search_type="similarity", search_kwargs={"k": 4},
#           embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
#     if mount_gdrive:
#         ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)
#
#     openai_docs_qa_chain = None
#     openai_api_qa_chain = None
#     local_docs_qa_chain = None
#     local_api_qa_chain = None
#
#     if load_gpt_model:
#         os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
#         open_ai_llm = startchat.get_openai_model_llm()
#         openai_docs_qa_chain, openai_api_qa_chain = get_openai_qa_chain(open_ai_llm, api_index_name_prefix,
#                                                                         docs_base_path,
#                                                                         docs_index_name_prefix, index_base_path,
#                                                                         search_kwargs, search_type,
#                                                                         is_openai_model=True)
#     if load_local_model:
#         llm = startchat.get_local_model_llm(
#             model_id=local_model_id,
#             use_4bit_quantization=use_4bit_quantization,
#             set_device_map=set_device_map,
#             max_new_tokens=max_new_tokens, device_map=device_map)
#
#         local_docs_qa_chain, local_api_qa_chain = get_local_qa_chain(llm, embedding_class, model_name,
#                                                                      api_index_name_prefix, docs_base_path,
#                                                                      docs_index_name_prefix, index_base_path,
#                                                                      search_kwargs, search_type, is_openai_model=False)
#
#     choices = ['Docs', 'API']
#     with gr.Blocks(theme="gradio/monochrome", mode="IWX CHATBOT", title="IWX CHATBOT") as demo:
#         chatbot_1 = gr.Chatbot(label="OPEN AI")
#         chatbot_2 = gr.Chatbot(label="IWX AI")
#         choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
#         msg = gr.Textbox(label="User Question")
#         clear = gr.ClearButton([msg, chatbot_1, chatbot_2, choice])
#
#         def user(user_message, history_1, history_2):
#             return gr.update(value="", interactive=False), history_1 + [[user_message, None]], history_2 + [
#                 [user_message, None]]
#
#         def bot_1(history_1, choice_selected):
#             query = get_question(history_1[-1][0], use_prompt, prompt, choice_selected)
#             if openai_api_qa_chain is not None and openai_docs_qa_chain is not None:
#                 if choice_selected == "API":
#                     bot_message = startchat.get_chat_gpt_result(openai_api_qa_chain, query)['result']
#                 else:
#                     bot_message = startchat.get_chat_gpt_result(openai_docs_qa_chain, query)['result']
#             else:
#                 bot_message = "Seams like open ai model is not loaded or not requested to give answer"
#             history_1[-1][1] = ""
#             for character in bot_message:
#                 history_1[-1][1] += character
#                 time.sleep(0.0005)
#                 yield history_1
#
#         def bot_2(history_1, history_2, choice_selected):
#             query = get_question(history_2[-1][0], use_prompt, prompt, choice_selected)
#             if local_api_qa_chain is not None and local_docs_qa_chain is not None:
#                 if choice_selected == "API":
#                     bot_message = startchat.get_local_model_result(local_api_qa_chain, query)['result']
#                 else:
#                     bot_message = startchat.get_local_model_result(local_docs_qa_chain, query)['result']
#             else:
#                 bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
#             record_answers(history_2[-1][0], history_1[-1][1], bot_message)
#             history_2[-1][1] = ""
#             for character in bot_message:
#                 history_2[-1][1] += character
#                 time.sleep(0.0005)
#                 yield history_2
#
#         response = msg.submit(user, [msg, chatbot_1, chatbot_2], [msg, chatbot_1, chatbot_2], queue=False)
#         response.then(bot_1, [chatbot_1, choice], chatbot_1)
#         response.then(bot_2, [chatbot_1, chatbot_2, choice], chatbot_2)
#         response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
#
#         demo.queue()
#         demo.launch(share=share_chat_ui, debug=debug)


# def start_qa_chain(load_gpt_model=False, load_local_model=True, local_model_id=constants.DEFAULT_MODEL_NAME,
#                    docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
#                    docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX,
#                    api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
#                    max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
#                    set_device_map=constants.SET_DEVICE_MAP,
#                    mount_gdrive=True,
#                    share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
#                    device_map=constants.DEFAULT_DEVICE_MAP, use_simple_llm_loader=False,
#                    embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain.prompts import PromptTemplate
#
#     api_prompt = PromptTemplate(template=constants.API_QUESTION_PROMPT,
#                                 input_variables=["context", "question"])
#     doc_prompt = PromptTemplate(template=constants.DOC_QUESTION_PROMPT,
#                                 input_variables=["context", "question"])
#
#     if mount_gdrive:
#         ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)
#
#     open_ai_llm = None
#     openai_docs_vector_store = None
#     openai_api_vector_store = None
#
#     local_llm = None
#     local_docs_vector_store = None
#     local_api_vector_store = None
#
#     if load_gpt_model:
#         os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
#         openai_embeddings = embeddings.get_openai_embeddings()
#         open_ai_llm = startchat.get_openai_model_llm()
#         openai_docs_vector_store, openai_api_vector_store = get_vector_stores(openai_embeddings, docs_base_path,
#                                                                               index_base_path, api_index_name_prefix,
#                                                                               docs_index_name_prefix,
#                                                                               is_openai_model=True)
#
#     if load_local_model:
#         local_llm = startchat.get_local_model_llm(
#             model_id=local_model_id,
#             use_4bit_quantization=use_4bit_quantization,
#             set_device_map=set_device_map,
#             max_new_tokens=max_new_tokens, device_map=device_map, use_simple_llm_loader=use_simple_llm_loader)
#
#         hf_embeddings = embeddings.get_embeddings(embedding_class, model_name)
#
#         local_docs_vector_store, local_api_vector_store = get_vector_stores(hf_embeddings, docs_base_path,
#                                                                             index_base_path, api_index_name_prefix,
#                                                                             docs_index_name_prefix,
#                                                                             is_openai_model=False)
#
#     choices = ['Docs', 'API']
#     with gr.Blocks(theme="gradio/monochrome", mode="IWX CHATBOT", title="IWX CHATBOT") as demo:
#         chatbot_1 = gr.Chatbot(label="OPEN AI")
#         chatbot_2 = gr.Chatbot(label="IWX AI")
#         choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
#         msg = gr.Textbox(label="User Question")
#         clear = gr.ClearButton([msg, chatbot_1, chatbot_2, choice])
#
#         def user(user_message, history_1, history_2):
#             return gr.update(value="", interactive=False), history_1 + [[user_message, None]], history_2 + [
#                 [user_message, None]]
#
#         def bot_1(history_1, choice_selected):
#             query = history_1[-1][0]
#             if open_ai_llm is not None:
#                 search_results = None
#                 openai_qa_chain = None
#                 if choice_selected == "API":
#                     search_results = openai_api_vector_store.similarity_search(query)
#                     openai_qa_chain = load_qa_chain(llm=open_ai_llm, chain_type="stuff", prompt=api_prompt)
#                 else:
#                     search_results = openai_docs_vector_store.similarity_search(query)
#                     openai_qa_chain = load_qa_chain(llm=open_ai_llm, chain_type="stuff", prompt=doc_prompt)
#
#                 if openai_qa_chain is not None and search_results is not None:
#                     result = openai_qa_chain({"input_documents": search_results, "question": query})
#                     bot_message = result["output_text"]
#                 else:
#                     bot_message = "No matching docs found on the vector store"
#             else:
#                 bot_message = "Seams like open ai model is not loaded or not requested to give answer"
#             history_1[-1][1] = ""
#             for character in bot_message:
#                 history_1[-1][1] += character
#                 time.sleep(0.0005)
#                 yield history_1
#
#         def bot_2(history_1, history_2, choice_selected):
#             query = history_2[-1][0]
#             if local_llm is not None:
#                 search_results = None
#                 local_qa_chain = None
#                 if choice_selected == "API":
#                     search_results = local_api_vector_store.similarity_search(query)
#                     local_qa_chain = load_qa_chain(llm=local_llm, chain_type="stuff", prompt=api_prompt)
#                 else:
#                     search_results = local_docs_vector_store.similarity_search(query)
#                     local_qa_chain = load_qa_chain(llm=local_llm, chain_type="stuff", prompt=doc_prompt)
#
#                 if local_qa_chain is not None and search_results is not None:
#                     result = local_qa_chain({"input_documents": search_results, "question": query})
#                     bot_message = result["output_text"]
#                 else:
#                     bot_message = "No matching docs found on the vector store"
#             else:
#                 bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
#             record_answers(history_2[-1][0], history_1[-1][1], bot_message)
#             history_2[-1][1] = ""
#             for character in bot_message:
#                 history_2[-1][1] += character
#                 time.sleep(0.0005)
#                 yield history_2
#
#         response = msg.submit(user, [msg, chatbot_1, chatbot_2], [msg, chatbot_1, chatbot_2], queue=False)
#         response.then(bot_1, [chatbot_1, choice], chatbot_1)
#         response.then(bot_2, [chatbot_1, chatbot_2, choice], chatbot_2)
#         response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
#
#         demo.queue()
#         demo.launch(share=share_chat_ui, debug=debug)

# def start_iwx(local_model_id=constants.DEFAULT_MODEL_NAME,
#               docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
#               docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX,
#               api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
#               max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
#               use_prompt=False, set_device_map=constants.SET_DEVICE_MAP,
#               mount_gdrive=True,
#               share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
#               device_map=constants.DEFAULT_DEVICE_MAP, search_type="similarity", search_kwargs={"k": 4},
#               embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large", use_queue=True,
#               use_simple_llm_loader=False, is_gptq_model=False, custom_quantization_config=None,
#               use_triton=False, use_safetensors=False, set_torch_dtype=False, torch_dtype=torch.bfloat16
#               ):
#     if mount_gdrive:
#         ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)
#
#     local_docs_qa_chain = None
#     local_api_qa_chain = None
#
#     llm = llmloader.load_llm(local_model_id, use_4bit_quantization=use_4bit_quantization, set_device_map=set_device_map,
#                              max_new_tokens=max_new_tokens, device_map=device_map,
#                              use_simple_llm_loader=use_simple_llm_loader, is_quantized_gptq_model=is_gptq_model,
#                              custom_quantiztion_config=custom_quantization_config, use_triton=use_triton,
#                              use_safetensors=use_safetensors, set_torch_dtype=set_torch_dtype, torch_dtype=torch_dtype)
#
#     local_docs_qa_chain, local_api_qa_chain = get_local_qa_chain(llm, embedding_class, model_name,
#                                                                  api_index_name_prefix, docs_base_path,
#                                                                  docs_index_name_prefix, index_base_path,
#                                                                  search_kwargs, search_type, is_openai_model=False)
#
#     choices = ['Docs', 'API']
#     data = [('Bad', '1'), ('Ok', '2'), ('Good', '3'), ('Very Good', '4'), ('Perfect', '5')]
#
#     def chatbot(choice_selected, message):
#         query = get_question(message, use_prompt, constants.API_QUESTION_PROMPT,
#                              choice_selected)
#         reference_docs = "Not populated"
#         if local_api_qa_chain is not None and local_docs_qa_chain is not None:
#             if choice_selected == "API":
#                 bot_message = startchat.get_local_model_result(local_api_qa_chain, query)['result']
#             else:
#                 bot_message = startchat.get_local_model_result(local_docs_qa_chain, query)['result']
#         else:
#             bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
#         print(bot_message)
#         print(reference_docs)
#         return bot_message, reference_docs
#
#     msg = gr.Textbox(label="User Question")
#     submit = gr.Button("Submit")
#     choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
#     output_textbox = gr.outputs.Textbox(label="IWX Bot")
#     output_textbox.show_copy_button = True
#     output_textbox.lines = 10
#     output_textbox.max_lines = 10
#
#     output_textbox1 = gr.outputs.Textbox(label="Reference Docs")
#     output_textbox1.lines = 2
#     output_textbox1.max_lines = 2
#
#     interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox, output_textbox1],
#                              theme="gradio/monochrome",
#                              title="IWX CHATBOT", allow_flagging="manual", flagging_callback=MysqlLogger(),
#                              flagging_options=data)
#
#     if use_queue:
#         interface.queue().launch(debug=debug, share=share_chat_ui)
#     else:
#         interface.launch(debug=debug, share=share_chat_ui)


# def ask(self, answer_type, query, similarity_search_k=4, api_prompt=None,
#         doc_prompt=None, code_prompt=None, summary_prompt=None):
#     if api_prompt is None:
#         api_prompt = self.api_prompt
#     if doc_prompt is None:
#         doc_prompt = self.doc_prompt
#     if code_prompt is None:
#         code_prompt = self.code_prompt
#     if summary_prompt is None:
#         summary_prompt = self.summary_prompt
#
#     reference_docs = ""
#     if self.llm_model is not None:
#         search_results = None
#         local_qa_chain = None
#         if answer_type == "API" or answer_type == "Code":
#             for api_vector_store in self.api_vector_stores:
#                 if search_results is None:
#                     search_results = api_vector_store.similarity_search(query, k=similarity_search_k)
#                 else:
#                     search_results = search_results + api_vector_store.similarity_search(query,
#                                                                                          k=similarity_search_k)
#             if answer_type == "API":
#                 local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=api_prompt)
#             else:
#                 local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=code_prompt)
#         elif answer_type == "Summary":
#             search_results = ingest.get_doc_from_text(query)
#             local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=summary_prompt)
#         else:
#             for doc_vector_store in self.doc_vector_stores:
#                 if search_results is None:
#                     search_results = doc_vector_store.similarity_search(query, k=similarity_search_k)
#                 else:
#                     search_results = search_results + doc_vector_store.similarity_search(queryk=similarity_search_k)
#             local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=doc_prompt)
#
#         if local_qa_chain is not None and search_results is not None:
#             result = local_qa_chain(
#                 {"input_documents": search_results, "question": query, "base_url": self.iwx_base_url})
#             bot_message = result["output_text"]
#             for doc in search_results:
#                 reference_docs = reference_docs + '\n' + str(doc.metadata.get('source'))
#         else:
#             bot_message = "No matching docs found on the vector store"
#     else:
#         bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
#
#     print(bot_message)
#     print(reference_docs)
#     return bot_message, reference_docs

# DEFAULT_PROMPT_FOR_API_2 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
#
# You are a REST API assistant working at Infoworks, but you are also an expert programmer.
# You are to complete the user request by composing a series of commands.
# Use the minimum number of commands required.
#
# The commands you have available are:
# | Command | Arguments | Description | Output Format |
# | --- | --- | --- | --- |
# | message | message | Send the user a message | null |
# | input | question | Ask the user for an input | null |
# | execute | APIRequest | execute an Infoworks v3 REST API request | null |
#
# always use the above mentioned commands and output json as shown in examples below, dont add anything extra to the answer.
#
# Example 1:
# [
#   {{
#     "command": "execute",
#     "arguments": {{
#       "type": "GET",
#       "url": "http://10.37.0.7:3001/v3/sources",
#       "headers": {{
#         "Content-Type": "application/json",
#         "Authorization": "Bearer {{refresh_token}}"
#       }},
#       "body": ""
#     }}
#   }}
# ]
#
# Example 2:
# Request: List all teradata sources
# Response:
# [
#   {{
#     "command": "execute",
#     "arguments": {{
#       "type": "GET",
#       "url": "{base_url}/v3/sources",
#       "headers": {{
#         "Content-Type": "application/json",
#         "Authorization": "Bearer {{refresh_token}}"
#       }},
#       "body": ""
#     }}
#   }}
# ]
#
# Example 3:
# User Request: create a teradata source with source name \"Teradata_sales\"
# Response:
# [
#   {{
#     "command": "input",
#     "arguments" : "Enter the source name"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the source type"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the source sub type"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the data lake path"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the environment id"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the storage id"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the data lake schema"
#   }},
#   {{
#     "command": "input",
#     "arguments" : "Enter the is_oem_connector"
#   }},
#   {{
#     "command": "execute",
#     "arguments": {{
#       "type": "POST"
#       "url": "{base_url}/api/v3/sources",
#       "headers": "{{\"Content-Type\": \"application/json\", \"Authorization\": \"Bearer {{refresh_token}}\"}}",
#       "body": {{
#         "name": "{{{{input_0}}}}",
#         "environment_id": "{{{{input_4}}}}",
#         "storage_id": "{{{{input_5}}",
#         "data_lake_schema": "{{{{input_6}}}}"
#         "data_lake_path": "{{{{input_3}}}}",
#         "type": "{{{{input_1}}}}",
#         "sub_type": "{{{{input_2}}}}",
#         "is_oem_connector": "{{{{input_7}}}}"
#       }}
#     }}
#   }}
# ]
#
# {context}
#
# IMPORTANT - Output the commands in JSON as an abstract syntax tree. Do not respond with any text that isn't part of a command.
# IMPORTANT - Do not explain yourself or Do not give Any Explanation
# IMPORTANT - You are an expert at generating commands and You can only generate commands.
# IMPORTANT - Do not assume any values of put or post or patch requests. always get the input from user any params.
# IMPORTANT - Infoworks is running on {base_url}
# IMPORTANT - Authenticate all execute commands using refresh_token assume user already has that information
#
# Understand the following request and generate the minimum set of commands as shown in examples above to complete it.
#
# Question: {question}
# """
#
#
# DEFAULT_PROMPT_FOR_API_3 = """Below is an instruction that describes a task. write a response that appropriately completes the request.
#
# ###INSTRUCTION:
#
# You are a REST API assistant working at Infoworks, but you are also an expert programmer.
# You are to complete the user request by composing a series of commands.
# Use the minimum number of commands required.
#
# IMPORTANT - Strictly follow below conditions while generating output.
# 1. Look for any value for the query or body parameters in the prompt before generating commands.
# 2. From the context provided below 'Request parameter' will give pipe delimited body parameters use that for Input Command.
# 3. From the context provided below 'Query parameter' will give pipe delimited query parameters use that for Input Command.
# 4. From the context provided below 'Method' will give you api call method GET/POST/PATCH.
# 5. For every POST and PATCH request ask user input for body parameters.
#
#
# IMPORTANT - The commands you have available are:
#
# | Command | Arguments  | Description                              |
# | ------- | ---------  | ---------------------------------------- |
# | Input   | question   | Ask input from user                      |
# | execute | APIRequest | execute an Infoworks v3 REST API request |
#
# IMPORTANT - Use these commands to Output the commands in JSON as an abstract syntax tree in one of the below format depending on <Method>:
#
# [
#   //Ask this for every parameter
#   {{
#     "command": "input",
#     "arguments": "Enter source id:"
#   }},
#   {{
#     "command": "input",
#     "arguments": "Enter table id:"
#   }},
#   {{
#     "command": "input",
#     "arguments": "Enter tags to add separated by comma (,):"
#   }},
#   {{
#     "command": "input",
#     "arguments": "Enter tags to remove separated by comma (,):"
#   }},
#   {{
#     "command": "input",
#     "arguments": "Enter is favorite (true/false):"
#   }},
#   {{
#     "command": "input",
#     "arguments": "Enter description:"
#   }},
#   {{
#     "command": "execute",
#     "arguments": {{
#       "type": "PUT",
#       "url": "{base_url}/v3/<path>",
#       "headers": {{
#         "Content-Type": "application/json",
#         "Authorization": "Bearer {{refresh_token}}"
#       }},
#       "body": {{
#         "tags_to_add": "{{{{input_2}}}}",
#         "tags_to_remove": "{{{{input_3}}}}",
#         "is_favorite": "{{{{input_4}}}}",
#         "description": "{{{{input_5}}}}"
#       }}
#     }}
#   }}
# ]
#
#
# IMPORTANT - Only use the above mentioned commands and output commands in JSON as an abstract syntax tree as shown in examples below.
# IMPORTANT - Do not respond with any text that isn't part of a command.
# IMPORTANT - Do not give Any kind of Explanation for your answer.
# IMPORTANT - You are an expert at generating commands and You can only generate commands.
# IMPORTANT - Do not assume any values of put or post or patch requests. always get the input from user any params.
# IMPORTANT - Infoworks is running on {base_url}.
# IMPORTANT - Authenticate all execute commands using refresh_token assume user already has that information.
#
#
# ###CONTEXT:
# {context}
#
#
# Question: {question}
#
# ###RESPONSE:
# """
#
# DEFAULT_PROMPT_FOR_SUMMARY_NEW = """Below is an instruction that describes a task. write a response that appropriately completes the request.
# Infoworks is running on {base_url}
# {question}
# {context}
# """
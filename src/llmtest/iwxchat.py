import gradio as gr
import time
import os
from getpass import getpass
from gradio import FlaggingCallback
from gradio.components import IOComponent
from typing import Any
from transformers import pipeline

from llmtest import constants, startchat, ingest, storage, embeddings, vectorstore

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings
)


def get_question(user_input, use_prompt, prompt):
    if use_prompt:
        return prompt + '\n' + user_input
    else:
        return user_input


def record_answers(query, open_ai_answer, local_model_answer):
    try:
        storage.insert_data(constants.USER_NAME, query, open_ai_answer, local_model_answer)
    except:
        print("Error while recording answers to db")
        pass


def start(load_gpt_model=True, load_local_model=True, local_model_id=constants.DEFAULT_MODEL_NAME,
          docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
          docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX, api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
          max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
          use_prompt=True, prompt=constants.QUESTION_PROMPT, set_device_map=constants.SET_DEVICE_MAP, mount_gdrive=True,
          share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
          device_map=constants.DEFAULT_DEVICE_MAP, search_type="similarity", search_kwargs={"k": 4},
          embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
    if mount_gdrive:
        ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)

    openai_docs_qa_chain = None
    openai_api_qa_chain = None
    local_docs_qa_chain = None
    local_api_qa_chain = None

    if load_gpt_model:
        os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
        open_ai_llm = startchat.get_openai_model_llm()
        openai_docs_qa_chain, openai_api_qa_chain = get_openai_qa_chain(open_ai_llm, api_index_name_prefix,
                                                                        docs_base_path,
                                                                        docs_index_name_prefix, index_base_path,
                                                                        search_kwargs, search_type,
                                                                        is_openai_model=True)
    if load_local_model:
        llm = startchat.get_local_model_llm(
            model_id=local_model_id,
            use_4bit_quantization=use_4bit_quantization,
            set_device_map=set_device_map,
            max_new_tokens=max_new_tokens, device_map=device_map)

        local_docs_qa_chain, local_api_qa_chain = get_local_qa_chain(llm, embedding_class, model_name,
                                                                     api_index_name_prefix, docs_base_path,
                                                                     docs_index_name_prefix, index_base_path,
                                                                     search_kwargs, search_type, is_openai_model=False)

    choices = ['Docs', 'API']
    with gr.Blocks(theme="gradio/monochrome", mode="IWX CHATBOT", title="IWX CHATBOT") as demo:
        chatbot_1 = gr.Chatbot(label="OPEN AI")
        chatbot_2 = gr.Chatbot(label="IWX AI")
        choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
        msg = gr.Textbox(label="User Question")
        clear = gr.ClearButton([msg, chatbot_1, chatbot_2, choice])

        def user(user_message, history_1, history_2):
            return gr.update(value="", interactive=False), history_1 + [[user_message, None]], history_2 + [
                [user_message, None]]

        def bot_1(history_1, choice_selected):
            query = get_question(history_1[-1][0], use_prompt, prompt)
            if openai_api_qa_chain is not None and openai_docs_qa_chain is not None:
                if choice_selected == "API":
                    bot_message = startchat.get_chat_gpt_result(openai_api_qa_chain, query)['result']
                else:
                    bot_message = startchat.get_chat_gpt_result(openai_docs_qa_chain, query)['result']
            else:
                bot_message = "Seams like open ai model is not loaded or not requested to give answer"
            history_1[-1][1] = ""
            for character in bot_message:
                history_1[-1][1] += character
                time.sleep(0.0005)
                yield history_1

        def bot_2(history_1, history_2, choice_selected):
            query = get_question(history_2[-1][0], use_prompt, prompt)
            if local_api_qa_chain is not None and local_docs_qa_chain is not None:
                if choice_selected == "API":
                    bot_message = startchat.get_local_model_result(local_api_qa_chain, query)['result']
                else:
                    bot_message = startchat.get_local_model_result(local_docs_qa_chain, query)['result']
            else:
                bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
            record_answers(history_2[-1][0], history_1[-1][1], bot_message)
            history_2[-1][1] = ""
            for character in bot_message:
                history_2[-1][1] += character
                time.sleep(0.0005)
                yield history_2

        response = msg.submit(user, [msg, chatbot_1, chatbot_2], [msg, chatbot_1, chatbot_2], queue=False)
        response.then(bot_1, [chatbot_1, choice], chatbot_1)
        response.then(bot_2, [chatbot_1, chatbot_2, choice], chatbot_2)
        response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

        demo.queue()
        demo.launch(share=share_chat_ui, debug=debug)


def get_openai_qa_chain(open_ai_llm, api_index_name_prefix, docs_base_path, docs_index_name_prefix, index_base_path,
                        search_kwargs,
                        search_type, is_openai_model):
    openai_embeddings = embeddings.get_openai_embeddings()

    docs_retriever, api_retriever = get_retrievers(openai_embeddings, docs_base_path, index_base_path,
                                                   api_index_name_prefix, docs_index_name_prefix, search_kwargs,
                                                   search_type, is_openai_model)

    openai_docs_qa_chain = startchat.get_openai_model_qa_chain(open_ai_llm, docs_retriever)
    openai_api_qa_chain = startchat.get_openai_model_qa_chain(open_ai_llm, api_retriever)

    return openai_docs_qa_chain, openai_api_qa_chain


def get_local_qa_chain(llm, embedding_class, model_name, api_index_name_prefix, docs_base_path, docs_index_name_prefix,
                       index_base_path, search_kwargs,
                       search_type, is_openai_model):
    hf_embeddings = embeddings.get_embeddings(embedding_class, model_name)

    docs_retriever, api_retriever = get_retrievers(hf_embeddings, docs_base_path, index_base_path,
                                                   api_index_name_prefix, docs_index_name_prefix, search_kwargs,
                                                   search_type, is_openai_model)

    openai_docs_qa_chain = startchat.get_openai_model_qa_chain(llm, docs_retriever)
    openai_api_qa_chain = startchat.get_openai_model_qa_chain(llm, api_retriever)

    return openai_docs_qa_chain, openai_api_qa_chain


def get_retrievers(model_embeddings, docs_base_path, index_base_path, api_index_name_prefix, docs_index_name_prefix,
                   search_kwargs,
                   search_type, is_openai_model):
    doc_vector_store, api_vector_store = get_vector_stores(model_embeddings, docs_base_path, index_base_path,
                                                           api_index_name_prefix, docs_index_name_prefix,
                                                           is_openai_model)

    docs_retriever = vectorstore.get_retriever_from_store(doc_vector_store, search_type=search_type,
                                                          search_kwargs=search_kwargs)

    api_retriever = vectorstore.get_retriever_from_store(api_vector_store, search_type=search_type,
                                                         search_kwargs=search_kwargs)
    return docs_retriever, api_retriever


def get_vector_stores(model_embeddings, docs_base_path, index_base_path, api_index_name_prefix, docs_index_name_prefix,
                      is_openai_model):
    if is_openai_model:
        index_base_path = index_base_path + "/openai/"
    else:
        index_base_path = index_base_path + "/hf/"
    doc_vector_store = vectorstore.get_vector_store(index_base_path=index_base_path,
                                                    index_name_prefix=docs_index_name_prefix,
                                                    docs_base_path=docs_base_path, embeddings=model_embeddings)
    api_vector_store = vectorstore.get_vector_store(index_base_path=index_base_path,
                                                    index_name_prefix=api_index_name_prefix,
                                                    docs_base_path=docs_base_path, embeddings=model_embeddings)
    return doc_vector_store, api_vector_store


def start_qa_chain(load_gpt_model=True, load_local_model=True, local_model_id=constants.DEFAULT_MODEL_NAME,
                   docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
                   docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX,
                   api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
                   max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
                   set_device_map=constants.SET_DEVICE_MAP,
                   mount_gdrive=True,
                   share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
                   device_map=constants.DEFAULT_DEVICE_MAP, use_simple_llm_loader=False,
                   embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate

    api_prompt = PromptTemplate(template=constants.DEFAULT_PROMPT_WITH_CONTEXT_API,
                                input_variables=["context", "question"])
    doc_prompt = PromptTemplate(template=constants.DEFAULT_PROMPT_WITH_CONTEXT_DOC,
                                input_variables=["context", "question"])

    if mount_gdrive:
        ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)

    open_ai_llm = None
    openai_docs_vector_store = None
    openai_api_vector_store = None

    local_llm = None
    local_docs_vector_store = None
    local_api_vector_store = None

    if load_gpt_model:
        os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
        openai_embeddings = embeddings.get_openai_embeddings()
        open_ai_llm = startchat.get_openai_model_llm()
        openai_docs_vector_store, openai_api_vector_store = get_vector_stores(openai_embeddings, docs_base_path,
                                                                              index_base_path, api_index_name_prefix,
                                                                              docs_index_name_prefix,
                                                                              is_openai_model=True)

    if load_local_model:
        local_llm = startchat.get_local_model_llm(
            model_id=local_model_id,
            use_4bit_quantization=use_4bit_quantization,
            set_device_map=set_device_map,
            max_new_tokens=max_new_tokens, device_map=device_map, use_simple_llm_loader=use_simple_llm_loader)

        hf_embeddings = embeddings.get_embeddings(embedding_class, model_name)

        local_docs_vector_store, local_api_vector_store = get_vector_stores(hf_embeddings, docs_base_path,
                                                                            index_base_path, api_index_name_prefix,
                                                                            docs_index_name_prefix,
                                                                            is_openai_model=False)

    choices = ['Docs', 'API']
    with gr.Blocks(theme="gradio/monochrome", mode="IWX CHATBOT", title="IWX CHATBOT") as demo:
        chatbot_1 = gr.Chatbot(label="OPEN AI")
        chatbot_2 = gr.Chatbot(label="IWX AI")
        choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
        msg = gr.Textbox(label="User Question")
        clear = gr.ClearButton([msg, chatbot_1, chatbot_2, choice])

        def user(user_message, history_1, history_2):
            return gr.update(value="", interactive=False), history_1 + [[user_message, None]], history_2 + [
                [user_message, None]]

        def bot_1(history_1, choice_selected):
            query = history_1[-1][0]
            if open_ai_llm is not None:
                search_results = None
                openai_qa_chain = None
                if choice_selected == "API":
                    search_results = openai_api_vector_store.similarity_search(query)
                    openai_qa_chain = load_qa_chain(llm=open_ai_llm, chain_type="stuff", prompt=api_prompt)
                else:
                    search_results = openai_docs_vector_store.similarity_search(query)
                    openai_qa_chain = load_qa_chain(llm=open_ai_llm, chain_type="stuff", prompt=doc_prompt)

                if openai_qa_chain is not None and search_results is not None:
                    result = openai_qa_chain({"input_documents": search_results, "question": query})
                    bot_message = result["output_text"]
                else:
                    bot_message = "No matching docs found on the vector store"
            else:
                bot_message = "Seams like open ai model is not loaded or not requested to give answer"
            history_1[-1][1] = ""
            for character in bot_message:
                history_1[-1][1] += character
                time.sleep(0.0005)
                yield history_1

        def bot_2(history_1, history_2, choice_selected):
            query = history_2[-1][0]
            if local_llm is not None:
                search_results = None
                local_qa_chain = None
                if choice_selected == "API":
                    search_results = local_api_vector_store.similarity_search(query)
                    local_qa_chain = load_qa_chain(llm=local_llm, chain_type="stuff", prompt=api_prompt)
                else:
                    search_results = local_docs_vector_store.similarity_search(query)
                    local_qa_chain = load_qa_chain(llm=local_llm, chain_type="stuff", prompt=doc_prompt)

                if local_qa_chain is not None and search_results is not None:
                    result = local_qa_chain({"input_documents": search_results, "question": query})
                    bot_message = result["output_text"]
                else:
                    bot_message = "No matching docs found on the vector store"
            else:
                bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
            record_answers(history_2[-1][0], history_1[-1][1], bot_message)
            history_2[-1][1] = ""
            for character in bot_message:
                history_2[-1][1] += character
                time.sleep(0.0005)
                yield history_2

        response = msg.submit(user, [msg, chatbot_1, chatbot_2], [msg, chatbot_1, chatbot_2], queue=False)
        response.then(bot_1, [chatbot_1, choice], chatbot_1)
        response.then(bot_2, [chatbot_1, chatbot_2, choice], chatbot_2)
        response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

        demo.queue()
        demo.launch(share=share_chat_ui, debug=debug)


class MysqlLogger(FlaggingCallback):

    def __init__(self):
        pass

    def setup(self, components: list[IOComponent], flagging_dir: str = None):
        self.components = components
        self.flagging_dir = flagging_dir
        print("here in setup")

    def flag(
            self,
            flag_data: list[Any],
            flag_option: str = "",
            username: str = None,
    ) -> int:
        data = []
        for component, sample in zip(self.components, flag_data):
            data.append(
                component.deserialize(
                    sample,
                    None,
                    None,
                )
            )
        data.append(flag_option)
        if len(data[1]) > 0 and len(data[2]) > 0:
            storage.insert_with_rating(constants.USER_NAME, data[0], data[1], data[2], data[3], data[4])
        else:
            print("no data to log")

        return 1


def start_iwx_only_chat(local_model_id=constants.DEFAULT_MODEL_NAME,
                        docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
                        docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX,
                        api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
                        max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
                        set_device_map=constants.SET_DEVICE_MAP,
                        mount_gdrive=True,
                        share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
                        device_map=constants.DEFAULT_DEVICE_MAP, use_simple_llm_loader=False,
                        embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate

    api_prompt = PromptTemplate(template=constants.DEFAULT_PROMPT_WITH_CONTEXT_API,
                                input_variables=["context", "question"])
    doc_prompt = PromptTemplate(template=constants.DEFAULT_PROMPT_WITH_CONTEXT_DOC,
                                input_variables=["context", "question"])

    if mount_gdrive:
        ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)

    local_llm = startchat.get_local_model_llm(
        model_id=local_model_id,
        use_4bit_quantization=use_4bit_quantization,
        set_device_map=set_device_map,
        max_new_tokens=max_new_tokens, device_map=device_map, use_simple_llm_loader=use_simple_llm_loader)

    hf_embeddings = embeddings.get_embeddings(embedding_class, model_name)

    local_docs_vector_store, local_api_vector_store = get_vector_stores(hf_embeddings, docs_base_path,
                                                                        index_base_path, api_index_name_prefix,
                                                                        docs_index_name_prefix,
                                                                        is_openai_model=False)

    choices = ['Docs', 'API']
    data = [('Bad', '1'), ('Ok', '2'), ('Good', '3'), ('Very Good', '4'), ('Perfect', '5')]

    def chatbot(choice_selected, message):
        query = message
        reference_docs = ""
        if local_llm is not None:
            search_results = None
            local_qa_chain = None
            if choice_selected == "API":
                search_results = local_api_vector_store.similarity_search(query)
                local_qa_chain = load_qa_chain(llm=local_llm, chain_type="stuff", prompt=api_prompt)
            else:
                search_results = local_docs_vector_store.similarity_search(query)
                local_qa_chain = load_qa_chain(llm=local_llm, chain_type="stuff", prompt=doc_prompt)

            if local_qa_chain is not None and search_results is not None:
                result = local_qa_chain({"input_documents": search_results, "question": query})
                bot_message = result["output_text"]
                for doc in search_results:
                    reference_docs = reference_docs + '\n' + str(doc.metadata.get('source'))
            else:
                bot_message = "No matching docs found on the vector store"
        else:
            bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
        # record_answers(query, "OPen AI Not configured", bot_message)
        print(bot_message)
        print(reference_docs)
        return bot_message, reference_docs

    msg = gr.Textbox(label="User Question")
    submit = gr.Button("Submit")
    choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
    output_textbox = gr.outputs.Textbox(label="IWX Bot")
    output_textbox.show_copy_button = True
    output_textbox.lines = 10
    output_textbox.max_lines = 10

    output_textbox1 = gr.outputs.Textbox(label="Reference Docs")
    output_textbox1.lines = 2
    output_textbox1.max_lines = 2

    interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox, output_textbox1],
                             theme="gradio/monochrome",
                             title="IWX CHATBOT", allow_flagging="manual", flagging_callback=MysqlLogger(),
                             flagging_options=data)
    interface.launch(debug=debug, share=share_chat_ui)

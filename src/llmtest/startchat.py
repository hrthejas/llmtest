import gradio as gr
import random
import time
from pprint import pprint
from IPython.display import display, Markdown
import os
from llmtest import llmloader, vectorstore, ingest, contants, pipeline_loader

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def get_embedding_retriever():
    return vectorstore.getRetrieverForChain(docs_base_path=contants.DOCS_BASE_PATH,
                                            index_base_path=contants.INDEX_BASE_PATH,
                                            index_name_prefix=contants.INDEX_NAME_PREFIX)


def load_local_model(retriever):
    llm = llmloader.getLLM(
        model_id=contants.DEFAULT_MODEL_NAME,
        use_4bit_quantization=contants.USE_4_BIT_QUANTIZATION,
        set_device_map=contants.SET_DEVICE_MAP,
        max_new_tokens=contants.MAX_NEW_TOKENS)

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def load_openai_model(retriever):
    os.environ["OPENAI_API_KEY"] = contants.OPEN_AI_API_KEY
    llm = ChatOpenAI(model_name=contants.OPEN_AI_MODEL_NAME, temperature=contants.OPEN_AI_TEMP,
                     max_tokens=contants.MAX_NEW_TOKENS)

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def get_local_model_result(qa, final_question):
    return qa(final_question)


def get_chat_gpt_result(qa, final_question):
    return qa(final_question)


def get_answers(local_qa, openai_qa, question):
    prompt = contants.QUESTION_PROMPT
    final_question = prompt + '\n' + question
    display(Markdown('*OPEN AI Result*'))
    pprint(get_chat_gpt_result(openai_qa, final_question)['result'])
    print('\n\n\n\n')
    display(Markdown('*local llm Result*'))
    pprint(get_local_model_result(local_qa, final_question)['result'])

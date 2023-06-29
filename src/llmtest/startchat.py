from IPython.display import display, Markdown
import os
from getpass import getpass

from IPython.display import display, Markdown
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from llmtest import llmloader, vectorstore, contants


def get_embedding_retriever(docs_base_path=contants.DOCS_BASE_PATH,
                            index_base_path=contants.INDEX_BASE_PATH,
                            index_name_prefix=contants.INDEX_NAME_PREFIX):
    return vectorstore.getRetrieverForChain(docs_base_path=docs_base_path,
                                            index_base_path=index_base_path,
                                            index_name_prefix=index_name_prefix)


def load_local_model(retriever, model_id=contants.DEFAULT_MODEL_NAME,
                     use_4bit_quantization=contants.USE_4_BIT_QUANTIZATION,
                     set_device_map=contants.SET_DEVICE_MAP,
                     max_new_tokens=contants.MAX_NEW_TOKENS):
    llm = llmloader.getLLM(
        model_id=contants.DEFAULT_MODEL_NAME,
        use_4bit_quantization=contants.USE_4_BIT_QUANTIZATION,
        set_device_map=contants.SET_DEVICE_MAP,
        max_new_tokens=contants.MAX_NEW_TOKENS)

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def load_openai_model(retriever):
    os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")
    llm = ChatOpenAI(model_name=contants.OPEN_AI_MODEL_NAME, temperature=contants.OPEN_AI_TEMP,
                     max_tokens=contants.MAX_NEW_TOKENS)

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def get_local_model_result(qa, final_question):
    return qa(final_question)


def get_chat_gpt_result(qa, final_question):
    return qa(final_question)


def get_answers(local_qa=None, openai_qa=None, question="list all environments in infoworks", use_prompt=True,
                prompt=contants.QUESTION_PROMPT):
    if use_prompt:
        final_question = prompt + '\n' + question
    else:
        final_question = question
    if openai_qa is not None:
        display(Markdown('*OPEN AI Result*'))
        print(get_chat_gpt_result(openai_qa, final_question)['result'])
        print('\n\n\n\n')
    if local_qa is not None:
        display(Markdown('*local llm Result*'))
        print(get_local_model_result(local_qa, final_question)['result'])

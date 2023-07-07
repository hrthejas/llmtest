import torch
from llmtest import llmloader, constants, vectorstore, ingest, iwxchat, embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings
)

doc_vector_stores = []
api_vector_stores = []
api_prompt = None
doc_prompt = None
llm_model = None
vector_embeddings = None


def init_prompt(api_prompt_template, doc_prompt_template):
    global api_prompt
    global doc_prompt
    api_prompt = PromptTemplate(template=api_prompt_template,
                                input_variables=["context", "question"])
    doc_prompt = PromptTemplate(template=doc_prompt_template,
                                input_variables=["context", "question"])


def init_embeddings(embedding_class, model_name):
    global vector_embeddings
    vector_embeddings = embeddings.get_embeddings(embedding_class, model_name)


def init_vector_store(mount_gdrive, gdrive_mount_base_bath, docs_base_path, index_base_path, api_index_name_prefix,
                      docs_index_name_prefix):
    if mount_gdrive:
        ingest.mountGoogleDrive(gdrive_mount_base_bath)

    global doc_vector_stores
    global api_vector_stores

    for prefix in docs_index_name_prefix:
        doc_vector_stores.append(vectorstore.get_vector_store(index_base_path=index_base_path,
                                                              index_name_prefix=prefix,
                                                              docs_base_path=docs_base_path,
                                                              embeddings=vector_embeddings))

    for prefix in api_index_name_prefix:
        api_vector_stores.append(vectorstore.get_vector_store(index_base_path=index_base_path,
                                                              index_name_prefix=prefix,
                                                              docs_base_path=docs_base_path,
                                                              embeddings=vector_embeddings))


def init_llm(model_id, max_new_tokens, use_4bit_quantization, set_device_map, device_map, use_simple_llm_loader,
             is_gptq_model, custom_quantization_config, use_safetensors, use_triton, set_torch_dtype, torch_dtype):
    global llm_model
    llm_model = llmloader.load_llm(model_id, use_4bit_quantization=use_4bit_quantization, set_device_map=set_device_map,
                                   max_new_tokens=max_new_tokens, device_map=device_map,
                                   use_simple_llm_loader=use_simple_llm_loader, is_quantized_gptq_model=is_gptq_model,
                                   custom_quantiztion_config=custom_quantization_config, use_triton=use_triton,
                                   use_safetensors=use_safetensors, set_torch_dtype=set_torch_dtype,
                                   torch_dtype=torch_dtype)


def init(model_id=constants.DEFAULT_MODEL_NAME, docs_base_path=constants.DOCS_BASE_PATH,
         index_base_path=constants.HF_INDEX_BASE_PATH, docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX,
         api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
         max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
         set_device_map=constants.SET_DEVICE_MAP,
         mount_gdrive=True,
         gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
         device_map=constants.DEFAULT_DEVICE_MAP, use_simple_llm_loader=False,
         embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large",
         is_gptq_model=False, custom_quantization_config=None, use_safetensors=False,
         use_triton=False, set_torch_dtype=False, torch_dtype=torch.bfloat16,
         api_prompt_template=constants.API_QUESTION_PROMPT, doc_prompt_template=constants.DOC_QUESTION_PROMPT):
    init_llm(model_id, max_new_tokens, use_4bit_quantization, set_device_map, device_map, use_simple_llm_loader,
             is_gptq_model, custom_quantization_config, use_safetensors, use_triton, set_torch_dtype, torch_dtype)

    init_embeddings(embedding_class, model_name)

    init_vector_store(mount_gdrive, gdrive_mount_base_bath, docs_base_path, index_base_path, api_index_name_prefix,
                      docs_index_name_prefix)

    init_prompt(api_prompt_template, doc_prompt_template)


def query_llm(answer_type, query, similarity_search_k=4):
    from langchain.chains.question_answering import load_qa_chain
    reference_docs = ""
    if llm_model is not None:
        search_results = None
        local_qa_chain = None
        if answer_type == "API":
            for api_vector_store in api_vector_stores:
                if search_results is None:
                    search_results = api_vector_store.similarity_search(query, k=similarity_search_k)
                else:
                    search_results = search_results + api_vector_store.similarity_search(query, k=similarity_search_k)
            local_qa_chain = load_qa_chain(llm=llm_model, chain_type="stuff", prompt=api_prompt)
        else:
            for doc_vector_store in doc_vector_stores:
                if search_results is None:
                    search_results = doc_vector_store.similarity_search(query, k=similarity_search_k)
                else:
                    search_results = search_results + doc_vector_store.similarity_search(queryk=similarity_search_k)
            local_qa_chain = load_qa_chain(llm=llm_model, chain_type="stuff", prompt=doc_prompt)

        if local_qa_chain is not None and search_results is not None:
            result = local_qa_chain({"input_documents": search_results, "question": query})
            bot_message = result["output_text"]
            for doc in search_results:
                reference_docs = reference_docs + '\n' + str(doc.metadata.get('source'))
        else:
            bot_message = "No matching docs found on the vector store"
    else:
        bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
    return bot_message, reference_docs

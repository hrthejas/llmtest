import torch
from llmtest import llmloader, constants, vectorstore, ingest, embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings
)


class ChatApp:
    doc_vector_stores = []
    api_vector_stores = []
    api_prompt = None
    doc_prompt = None
    llm_model = None
    vector_embeddings = None

    app_args = ["model_id", "docs_base_path", "index_base_path", "docs_index_name_prefix", "api_index_name_prefix",
                "max_new_tokens", "use_4bit_quantization", "set_device_map", "mount_gdrive", "gdrive_mount_base_bath",
                "device_map", "use_simple_llm_loader", "embedding_class", "model_name", "is_gptq_model",
                "custom_quantization_config", "use_safetensors", "use_triton", "set_torch_dtype", "torch_dtype",
                "api_prompt_template", "doc_prompt_template"]

    model_id = constants.DEFAULT_MODEL_NAME
    docs_base_path = constants.DOCS_BASE_PATH
    index_base_path = constants.HF_INDEX_BASE_PATH
    docs_index_name_prefix = constants.DOC_INDEX_NAME_PREFIX
    api_index_name_prefix = constants.API_INDEX_NAME_PREFIX
    max_new_tokens = constants.MAX_NEW_TOKENS
    use_4bit_quantization = constants.USE_4_BIT_QUANTIZATION
    set_device_map = constants.SET_DEVICE_MAP
    mount_gdrive = True
    gdrive_mount_base_bath = constants.GDRIVE_MOUNT_BASE_PATH
    device_map = constants.DEFAULT_DEVICE_MAP
    use_simple_llm_loader = False
    embedding_class = HuggingFaceInstructEmbeddings
    model_name = "hkunlp/instructor-large"
    is_gptq_model = False
    custom_quantization_config = None
    use_safetensors = False
    use_triton = False
    set_torch_dtype = False
    torch_dtype = torch.bfloat16
    api_prompt_template = constants.API_QUESTION_PROMPT
    doc_prompt_template = constants.DOC_QUESTION_PROMPT

    def __getitem__(self, item):
        return item

    def __int__(self, **kwargs):
        if len(kwargs) > 0:
            valid_kwargs = {name: kwargs.pop(name) for name in self.app_args if name in kwargs}
            for key, value in valid_kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        self.llm_model = llmloader.load_llm(self.model_id, use_4bit_quantization=self.use_4bit_quantization,
                                            set_device_map=self.set_device_map,
                                            max_new_tokens=self.max_new_tokens, device_map=self.device_map,
                                            use_simple_llm_loader=self.use_simple_llm_loader,
                                            is_quantized_gptq_model=self.is_gptq_model,
                                            custom_quantiztion_config=self.custom_quantization_config,
                                            use_triton=self.use_triton,
                                            use_safetensors=self.use_safetensors, set_torch_dtype=self.set_torch_dtype,
                                            torch_dtype=self.torch_dtype)

        self.vector_embeddings = embeddings.get_embeddings(self.embedding_class, self.model_name)

        if self.mount_gdrive:
            ingest.mountGoogleDrive(self.gdrive_mount_base_bath)

        for prefix in self.docs_index_name_prefix:
            self.doc_vector_stores.append(vectorstore.get_vector_store(index_base_path=self.index_base_path,
                                                                       index_name_prefix=prefix,
                                                                       docs_base_path=self.docs_base_path,
                                                                       embeddings=self.vector_embeddings))

        for prefix in self.api_index_name_prefix:
            self.api_vector_stores.append(vectorstore.get_vector_store(index_base_path=self.index_base_path,
                                                                       index_name_prefix=prefix,
                                                                       docs_base_path=self.docs_base_path,
                                                                       embeddings=self.vector_embeddings))

        self.api_prompt = PromptTemplate(template=self.api_prompt_template,
                                         input_variables=["context", "question"])

        self.doc_prompt = PromptTemplate(template=self.doc_prompt_template,
                                         input_variables=["context", "question"])

    def query_llm(self, answer_type, query, similarity_search_k=4):
        reference_docs = ""
        if self.llm_model is not None:
            search_results = None
            local_qa_chain = None
            if answer_type == "API":
                for api_vector_store in self.api_vector_stores:
                    if search_results is None:
                        search_results = api_vector_store.similarity_search(query, k=similarity_search_k)
                    else:
                        search_results = search_results + api_vector_store.similarity_search(query,
                                                                                             k=similarity_search_k)
                local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=self.api_prompt)
            else:
                for doc_vector_store in self.doc_vector_stores:
                    if search_results is None:
                        search_results = doc_vector_store.similarity_search(query, k=similarity_search_k)
                    else:
                        search_results = search_results + doc_vector_store.similarity_search(queryk=similarity_search_k)
                local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=self.doc_prompt)

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

import torch
import gradio as gr
from llmtest import llmloader, constants, vectorstore, ingest, embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings
)
from langchain.chat_models import ChatOpenAI

from llmtest.MysqlLogger import MysqlLogger


class IWXBot:
    doc_vector_stores = []
    api_vector_stores = []
    api_prompt = None
    doc_prompt = None
    code_prompt = None
    summary_prompt = None
    llm_model = None
    summary_llm_model = None
    vector_embeddings = None

    app_args = ["model_id", "docs_base_path", "index_base_path", "docs_index_name_prefix", "api_index_name_prefix",
                "max_new_tokens", "use_4bit_quantization", "set_device_map", "mount_gdrive", "gdrive_mount_base_bath",
                "device_map", "use_simple_llm_loader", "embedding_class", "model_name", "is_gptq_model",
                "custom_quantization_config", "use_safetensors", "use_triton", "set_torch_dtype", "torch_dtype",
                "api_prompt_template", "doc_prompt_template", "code_prompt_template"]

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
    summary_model_name = constants.OPEN_AI_MODEL_NAME
    summary_temperature = constants.OPEN_AI_TEMP
    is_gptq_model = False
    custom_quantization_config = None
    use_safetensors = False
    use_triton = False
    set_torch_dtype = False
    torch_dtype = torch.bfloat16
    api_prompt_template = constants.API_QUESTION_PROMPT
    doc_prompt_template = constants.DOC_QUESTION_PROMPT
    code_prompt_template = constants.DEFAULT_PROMPT_FOR_CODE
    summary_prompt_template = constants.DEFAULT_PROMPT_FOR_SUMMARY

    def __getitem__(self, item):
        return item

    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            valid_kwargs = {name: kwargs.pop(name) for name in self.app_args if name in kwargs}
            for key, value in valid_kwargs.items():
                if hasattr(self, key):
                    print("Setting attribute " + key)
                    setattr(self, key, value)

        if self.mount_gdrive:
            ingest.mountGoogleDrive(self.gdrive_mount_base_bath)

        pass

    def initialize_chat(self):

        self.vector_embeddings = embeddings.get_embeddings(self.embedding_class, self.model_name)

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

        self.code_prompt = PromptTemplate(template=self.code_prompt_template,
                                          input_variables=["context", "question"])

        self.summary_prompt = PromptTemplate(template=self.summary_prompt_template,
                                             input_variables=["context", "question"])

        self.llm_model = llmloader.load_llm(self.model_id, use_4bit_quantization=self.use_4bit_quantization,
                                            set_device_map=self.set_device_map,
                                            max_new_tokens=self.max_new_tokens, device_map=self.device_map,
                                            use_simple_llm_loader=self.use_simple_llm_loader,
                                            is_quantized_gptq_model=self.is_gptq_model,
                                            custom_quantiztion_config=self.custom_quantization_config,
                                            use_triton=self.use_triton,
                                            use_safetensors=self.use_safetensors, set_torch_dtype=self.set_torch_dtype,
                                            torch_dtype=self.torch_dtype)
        self.summary_llm_model = ChatOpenAI(model_name=self.summary_model_name, temperature=self.temperature,
                                            max_tokens=self.max_new_tokens)
        print("Loaded all prompts")
        print("Init complete")
        pass

    def ask(self, answer_type, query, similarity_search_k=4, api_prompt=None,
            doc_prompt=None, code_prompt=None, summary_prompt=None):

        if api_prompt is None:
            api_prompt = self.api_prompt
        if doc_prompt is None:
            doc_prompt = self.doc_prompt
        if code_prompt is None:
            code_prompt = self.code_prompt
        if summary_prompt is None:
            summary_prompt = self.summary_prompt

        reference_docs = ""
        if self.llm_model is not None:
            search_results = None
            local_qa_chain = None
            if answer_type == "API" or answer_type == "Code":
                for api_vector_store in self.api_vector_stores:
                    if search_results is None:
                        search_results = api_vector_store.similarity_search(query, k=similarity_search_k)
                    else:
                        search_results = search_results + api_vector_store.similarity_search(query,
                                                                                             k=similarity_search_k)
                if answer_type == "API":
                    local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=api_prompt)
                else:
                    local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=code_prompt)
            elif answer_type == "Summary":
                search_results = ingest.get_doc_from_text(query)
                local_qa_chain = load_qa_chain(llm=self.summary_llm_model, chain_type="stuff", prompt=summary_prompt)
            else:
                for doc_vector_store in self.doc_vector_stores:
                    if search_results is None:
                        search_results = doc_vector_store.similarity_search(query, k=similarity_search_k)
                    else:
                        search_results = search_results + doc_vector_store.similarity_search(queryk=similarity_search_k)
                local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=doc_prompt)

            if local_qa_chain is not None and search_results is not None:
                result = local_qa_chain({"input_documents": search_results, "question": query})
                bot_message = result["output_text"]
                for doc in search_results:
                    reference_docs = reference_docs + '\n' + str(doc.metadata.get('source'))
            else:
                bot_message = "No matching docs found on the vector store"
        else:
            bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"

        print(bot_message)
        print(reference_docs)
        return bot_message, reference_docs

    def ask_with_prompt(self, answer_type, query, similarity_search_k=4,
                        api_prompt_template=api_prompt_template,
                        doc_prompt_template=doc_prompt_template,
                        code_prompt_template=code_prompt_template,
                        summary_prompt_template=summary_prompt_template):

        api_prompt = PromptTemplate(template=api_prompt_template,
                                    input_variables=["context", "question"])

        doc_prompt = PromptTemplate(template=doc_prompt_template,
                                    input_variables=["context", "question"])

        code_prompt = PromptTemplate(template=code_prompt_template,
                                     input_variables=["context", "question"])

        summary_prompt = PromptTemplate(template=summary_prompt_template,
                                        input_variables=["context", "question"])

        return self.ask(answer_type, query, similarity_search_k, api_prompt, doc_prompt, code_prompt, summary_prompt)

    def start_chat(self, debug=True, use_queue=False, share_ui=True, similarity_search_k=1, record_feedback=True,
                   api_prompt_template=constants.API_QUESTION_PROMPT,
                   doc_prompt_template=constants.DOC_QUESTION_PROMPT,
                   code_prompt_template=constants.DEFAULT_PROMPT_FOR_CODE,
                   summary_prompt_template=constants.DEFAULT_PROMPT_FOR_SUMMARY,
                   add_summary_answer_type=False):
        if add_summary_answer_type:
            choices = ['API', 'Docs', 'Code', 'Summary']
        else:
            choices = ['API', 'Docs', 'Code']
        data = [('Bad', '1'), ('Ok', '2'), ('Good', '3'), ('Very Good', '4'), ('Perfect', '5')]

        def chatbot(choice_selected, message):
            return self.ask_with_prompt(choice_selected, message, similarity_search_k=similarity_search_k,
                                        api_prompt_template=api_prompt_template,
                                        doc_prompt_template=doc_prompt_template,
                                        code_prompt_template=code_prompt_template,
                                        summary_prompt_template=summary_prompt_template)

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

        if record_feedback:
            interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox, output_textbox1],
                                     theme="darkhuggingface",
                                     title="IWX CHATBOT", allow_flagging="manual", flagging_callback=MysqlLogger(),
                                     flagging_options=data)
        else:
            interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox, output_textbox1],
                                     theme="darkhuggingface",
                                     title="IWX CHATBOT", allow_flagging="never")
        if use_queue:
            interface.queue().launch(debug=debug, share=share_ui)
        else:
            interface.launch(debug=debug, share=share_ui)

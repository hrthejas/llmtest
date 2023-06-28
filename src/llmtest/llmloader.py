import torch
import os
from transformers import BitsAndBytesConfig
from enum import Enum
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    AutoModelForQuestionAnswering,
    GenerationConfig
    )

from langchain import (
    HuggingFaceHub,
    HuggingFacePipeline
    )

from langchain.chains import RetrievalQA
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings
    )
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import (
    Chroma,
    FAISS,
    ElasticVectorSearch
    )
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
    )
from langchain.document_loaders import (
    DirectoryLoader,
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredAPIFileLoader,
)

class IndexType(Enum):
    FAISS_INDEX = 1
    CHROMA_INDEX = 2
    ELASTIC_SEARCH_INDEX = 3

def getTokenizer(model_id,tokenizer_class=AutoTokenizer):
    tokenizer = tokenizer_class.from_pretrained(model_id)
    return tokenizer;

def getQuantizedModel(model_id,model_class=AutoModelForCausalLM):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
    model = model_class.from_pretrained(model_id,device_map="auto",quantization_config=quantization_config,trust_remote_code=True)
    return model

def getNonQuantizedModel(model_id,model_class=AutoModelForCausalLM):
    model = model_class.from_pretrained(model_id,trust_remote_code=True)
    return model

def loadModel(model_id,use_4bit_quantization=False,model_class=AutoModelForCausalLM):
  if use_4bit_quantization == True :
      return getQuantizedModel(model_id,model_class)
  else:
      return getNonQuantizedModel(model_id,model_class)
  

def getPipeLIneWithDeviceMap(
            model,
            tokenizer,
            task="text-generation",
            use_cache=True,
            device_map="auto",
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            max_new_tokens=256,
            ):
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_cache=use_cache,
        device_map=device_map,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens
    )
    
def getPipeLIneWithoutDeviceMap(
            model,
            tokenizer,
            task="text-generation",
            use_cache=True,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            max_new_tokens=256,
            ):
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_cache=use_cache,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens
    )

def mountGoogleDrive(mount_location="/content/drive") :
    from google.colab import drive
    # This will prompt for authorization.
    drive.mount(mount_location)
    
def getEmbeddings(embedding_class=HuggingFaceInstructEmbeddings,model_name="hkunlp/instructor-large") : 
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    return embedding_class(model_name=model_name)

def getHTMLDocs(html_api_docs_path,chunk_size=1000,chunk_overlap=100) : 
    loader = DirectoryLoader(html_api_docs_path, recursive=True, glob="**/*.html", loader_cls=UnstructuredHTMLLoader)
    html_docs = loader.load()
    html_text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    html_text = html_text_splitter.split_documents(html_docs)
    return html_text

def getMarkDownDocs(md_docs_path,chunk_size=1000,chunk_overlap=100) : 
    loader = DirectoryLoader(md_docs_path, recursive=True, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    md_docs = loader.load()
    md_text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    md_text = md_text_splitter.split_documents(md_docs)
    return md_text

def faissRetriever(embeddings,docs_content,index_base_path,index_name_prefix,is_overwrite=False,search_type="similarity", search_kwargs={"k":1}) :
    faiss_vector_store_path = index_base_path + "/faiss/" + index_name_prefix + "/"
    if is_overwrite == True and len(docs_content) > 0 :
        vector_store = FAISS.from_documents(docs_content, embeddings)
        vector_store.save_local(folder_path=faiss_vector_store_path,index_name=index_name_prefix) 
        return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    else :
        vector_store = FAISS.load_local(folder_path=faiss_vector_store_path,embeddings=embeddings,index_name=index_name_prefix)
        return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

def chromaRetriever(embeddings,docs_content,index_base_path,index_name_prefix,is_overwrite=False,search_type="similarity", search_kwargs={"k":1}) :
    from chromadb.config import Settings
    chroma_vector_store_path = index_base_path + "/chroma/" + index_name_prefix

    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=chroma_vector_store_path,
        anonymized_telemetry=False
        )
    
    if is_overwrite == True and len(docs_content) > 0 :
        db = Chroma.from_documents(docs_content, embeddings, persist_directory=chroma_vector_store_path, client_settings=CHROMA_SETTINGS)
        db.persist()
        return db.as_retriever(search_kwargs=search_kwargs)
    else :
        db = Chroma(persist_directory=chroma_vector_store_path, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        return db.as_retriever(search_kwargs=search_kwargs)

def elasticRetriever(embeddings,docs_content,index_base_path,index_name_prefix,is_overwrite=False,search_type="similarity", search_kwargs={"k":1}) :
    from langchain import ElasticVectorSearch
    endpoint = os.environ.get('elastic_search_endpoint') #806b9000925b41b296b030ca49a1fd9a.es.us-central1.gcp.cloud.es.io
    username = os.environ.get('elastic_search_username') #elastic
    password = os.environ.get('elastic_search_password') #8TjuUPdWLr7of2jpDxU5V1Vi1
    url = f"https://{username}:{password}@{endpoint}:443"
    db = ElasticVectorSearch(embedding=embeddings, elasticsearch_url=url, index_name=index_name_prefix)
    elastic_index = db.from_documents(docs_content, embeddings,elasticsearch_url=url)
    return elastic_index.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

def getRetrieverForChain(docs_base_path,index_base_path,index_name_prefix,embedding_class=HuggingFaceInstructEmbeddings,model_name="hkunlp/instructor-large",index_type=IndexType.FAISS_INDEX,is_overwrite=False,read_html_docs=True,read_md_docs=True,chunk_size=1000,chunk_overlap=100,search_type="similarity", search_kwargs={"k":1}) :
    embeddings = getEmbeddings(embedding_class=embedding_class,model_name=model_name)
    all_docs = list()
    if is_overwrite == True :
        # Reading Docs from the path 
        if read_html_docs == True :
            html_docs = getHTMLDocs(docs_base_path,chunk_overlap=chunk_overlap,chunk_size=chunk_size)
            all_docs = all_docs + html_docs
        if read_md_docs == True :
            md_docs = getMarkDownDocs(docs_base_path,chunk_overlap=chunk_overlap,chunk_size=chunk_size)
            all_docs = all_docs + md_docs
            
    if index_type == IndexType.FAISS_INDEX :
        return faissRetriever(embeddings,all_docs,index_base_path,index_name_prefix,is_overwrite=False,search_type="similarity", search_kwargs={"k":1})
    elif index_type == IndexType.CHROMA_INDEX :
        return chromaRetriever(embeddings,all_docs,index_base_path,index_name_prefix,is_overwrite=False,search_type="similarity", search_kwargs={"k":1})
    elif index_type == IndexType.ELASTIC_SEARCH_INDEX :
        return chromaRetriever(embeddings,all_docs,index_base_path,index_name_prefix,is_overwrite=False,search_type="similarity", search_kwargs={"k":1})
    else :
        raise Exception("Sorry, Unknown index_type : " + index_type)

def getLLM(
    model_id,
    use_4bit_quantization=False,
    model_class=AutoModelForCausalLM,
    tokenizer_class=AutoTokenizer,
    task="text-generation",
    use_cache=True,
    device_map="auto",
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    max_new_tokens=256,
    set_device_map=False
    ):
    model = loadModel(model_id,use_4bit_quantization,model_class)
    tokenizer = getTokenizer(model_id,tokenizer_class)
    if set_device_map == True :
        pipeline = getPipeLIneWithDeviceMap(
            model,
            tokenizer,
            task="text-generation",
            use_cache=True,
            device_map="auto",
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            max_new_tokens=256,
            )
        return HuggingFacePipeline(pipeline=pipeline)
    else :
        pipeline = getPipeLIneWithoutDeviceMap(
            model,
            tokenizer,
            task="text-generation",
            use_cache=True,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            max_new_tokens=256,
            )
        return HuggingFacePipeline(pipeline=pipeline)

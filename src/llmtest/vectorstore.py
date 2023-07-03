import os
from llmtest import constants, indextype, ingest
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import (
    Chroma,
    FAISS,
    ElasticVectorSearch
)

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings
)


def getEmbeddings(embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    return embedding_class(model_name=model_name)


def get_retriever_from_store(store, search_type="similarity", search_kwargs={"k": 4}):
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def faiss_db(embeddings, docs_content, index_base_path, index_name_prefix, is_overwrite=False):
    faiss_vector_store_path = index_base_path + "/faiss/" + index_name_prefix + "/"
    vector_store = None
    if is_overwrite == True and len(docs_content) > 0:
        vector_store = FAISS.from_documents(docs_content, embeddings)
        vector_store.save_local(folder_path=faiss_vector_store_path, index_name=index_name_prefix)
    else:
        vector_store = FAISS.load_local(folder_path=faiss_vector_store_path, embeddings=embeddings,
                                        index_name=index_name_prefix)
    return vector_store


def chroma_db(embeddings, docs_content, index_base_path, index_name_prefix, is_overwrite=False):
    from chromadb.config import Settings
    chroma_vector_store_path = index_base_path + "/chroma/" + index_name_prefix

    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=chroma_vector_store_path,
        anonymized_telemetry=False
    )
    db = None
    if is_overwrite and len(docs_content) > 0:
        db = Chroma.from_documents(docs_content, embeddings, persist_directory=chroma_vector_store_path,
                                   client_settings=CHROMA_SETTINGS)
        db.persist()
    else:
        db = Chroma(persist_directory=chroma_vector_store_path, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS)
    return db


def elastic_db(embeddings, docs_content, index_base_path, index_name_prefix, is_overwrite=False):
    from langchain import ElasticVectorSearch
    endpoint = os.environ.get(
        'elastic_search_endpoint')  # 806b9000925b41b296b030ca49a1fd9a.es.us-central1.gcp.cloud.es.io
    username = os.environ.get('elastic_search_username')  # elastic
    password = os.environ.get('elastic_search_password')  # 8TjuUPdWLr7of2jpDxU5V1Vi1
    url = f"https://{username}:{password}@{endpoint}:443"
    db = ElasticVectorSearch(embedding=embeddings, elasticsearch_url=url, index_name=index_name_prefix)
    elastic_index = db.from_documents(docs_content, embeddings, elasticsearch_url=url)
    return elastic_index


def get_retriever_for_chain(docs_base_path, index_base_path, index_name_prefix,
                            embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large",
                            index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
                            read_md_docs=True,
                            chunk_size=1000, chunk_overlap=100, search_type="similarity", search_kwargs={"k": 1}):
    embeddings = getEmbeddings(embedding_class=embedding_class, model_name=model_name)
    all_docs = list()
    if is_overwrite == True:
        # Reading Docs from the path
        if read_html_docs == True:
            html_docs = ingest.getHTMLDocs(docs_base_path, chunk_overlap=chunk_overlap, chunk_size=chunk_size)
            all_docs = all_docs + html_docs
        if read_md_docs == True:
            md_docs = ingest.getMarkDownDocs(docs_base_path, chunk_overlap=chunk_overlap, chunk_size=chunk_size)
            all_docs = all_docs + md_docs

    vectore_store = None

    if index_type == indextype.IndexType.FAISS_INDEX:
        vectore_store = faiss_db(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False)
    elif index_type == indextype.IndexType.CHROMA_INDEX:
        vectore_store = chroma_db(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False)
    elif index_type == indextype.IndexType.ELASTIC_SEARCH_INDEX:
        vectore_store = elastic_db(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False)

    if vectore_store is not None:
        get_retriever_from_store(vectore_store, search_type=search_type, search_kwargs=search_kwargs)
    else:
        raise Exception("Sorry, Unknown index_type : ")


def get_retriever_for_openai_chain(docs_base_path, index_base_path, index_name_prefix,
                                   index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
                                   read_md_docs=True,
                                   chunk_size=1000, chunk_overlap=100,search_type="similarity", search_kwargs={"k": 4}):
    embeddings = OpenAIEmbeddings()
    all_docs = list()
    if is_overwrite == True:
        # Reading Docs from the path
        if read_html_docs == True:
            html_docs = ingest.getHTMLDocs(docs_base_path, chunk_overlap=chunk_overlap, chunk_size=chunk_size)
            all_docs = all_docs + html_docs
        if read_md_docs == True:
            md_docs = ingest.getMarkDownDocs(docs_base_path, chunk_overlap=chunk_overlap, chunk_size=chunk_size)
            all_docs = all_docs + md_docs

    vectore_store = None

    if index_type == indextype.IndexType.FAISS_INDEX:
        vectore_store = faiss_db(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False)
    elif index_type == indextype.IndexType.CHROMA_INDEX:
        vectore_store = chroma_db(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False)
    elif index_type == indextype.IndexType.ELASTIC_SEARCH_INDEX:
        vectore_store = elastic_db(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False)

    if vectore_store is not None:
        get_retriever_from_store(vectore_store, search_type=search_type, search_kwargs=search_kwargs)
    else:
        raise Exception("Sorry, Unknown index_type : ")

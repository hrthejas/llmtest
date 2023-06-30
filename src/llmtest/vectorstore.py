import os
from llmtest import contants, indextype, ingest
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


def faissRetriever(embeddings, docs_content, index_base_path, index_name_prefix, is_overwrite=False,
                   search_type="similarity", search_kwargs={"k": 1}):
    faiss_vector_store_path = index_base_path + "/faiss/" + index_name_prefix + "/"
    if is_overwrite == True and len(docs_content) > 0:
        vector_store = FAISS.from_documents(docs_content, embeddings)
        vector_store.save_local(folder_path=faiss_vector_store_path, index_name=index_name_prefix)
        return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    else:
        vector_store = FAISS.load_local(folder_path=faiss_vector_store_path, embeddings=embeddings,
                                        index_name=index_name_prefix)
        return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def chromaRetriever(embeddings, docs_content, index_base_path, index_name_prefix, is_overwrite=False,
                    search_type="similarity", search_kwargs={"k": 1}):
    from chromadb.config import Settings
    chroma_vector_store_path = index_base_path + "/chroma/" + index_name_prefix

    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=chroma_vector_store_path,
        anonymized_telemetry=False
    )

    if is_overwrite == True and len(docs_content) > 0:
        db = Chroma.from_documents(docs_content, embeddings, persist_directory=chroma_vector_store_path,
                                   client_settings=CHROMA_SETTINGS)
        db.persist()
        return db.as_retriever(search_kwargs=search_kwargs)
    else:
        db = Chroma(persist_directory=chroma_vector_store_path, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS)
        return db.as_retriever(search_kwargs=search_kwargs)


def elasticRetriever(embeddings, docs_content, index_base_path, index_name_prefix, is_overwrite=False,
                     search_type="similarity", search_kwargs={"k": 1}):
    from langchain import ElasticVectorSearch
    endpoint = os.environ.get(
        'elastic_search_endpoint')  # 806b9000925b41b296b030ca49a1fd9a.es.us-central1.gcp.cloud.es.io
    username = os.environ.get('elastic_search_username')  # elastic
    password = os.environ.get('elastic_search_password')  # 8TjuUPdWLr7of2jpDxU5V1Vi1
    url = f"https://{username}:{password}@{endpoint}:443"
    db = ElasticVectorSearch(embedding=embeddings, elasticsearch_url=url, index_name=index_name_prefix)
    elastic_index = db.from_documents(docs_content, embeddings, elasticsearch_url=url)
    return elastic_index.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def getRetrieverForChain(docs_base_path, index_base_path, index_name_prefix,
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

    if index_type == indextype.IndexType.FAISS_INDEX:
        return faissRetriever(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False,
                              search_type="similarity", search_kwargs={"k": 1})
    elif index_type == indextype.IndexType.CHROMA_INDEX:
        return chromaRetriever(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False,
                               search_type="similarity", search_kwargs={"k": 1})
    elif index_type == indextype.IndexType.ELASTIC_SEARCH_INDEX:
        return chromaRetriever(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False,
                               search_type="similarity", search_kwargs={"k": 1})
    else:
        raise Exception("Sorry, Unknown index_type : ")


def getRetrieverForOpenAIChain(docs_base_path, index_base_path, index_name_prefix,
                               index_type=indextype.IndexType.FAISS_INDEX, is_overwrite=False, read_html_docs=True,
                               read_md_docs=True,
                               chunk_size=1000, chunk_overlap=100):
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

    if index_type == indextype.IndexType.FAISS_INDEX:
        return faissRetriever(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False,
                              search_type="similarity", search_kwargs={"k": 1})
    elif index_type == indextype.IndexType.CHROMA_INDEX:
        return chromaRetriever(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False,
                               search_type="similarity", search_kwargs={"k": 1})
    elif index_type == indextype.IndexType.ELASTIC_SEARCH_INDEX:
        return chromaRetriever(embeddings, all_docs, index_base_path, index_name_prefix, is_overwrite=False,
                               search_type="similarity", search_kwargs={"k": 1})
    else:
        raise Exception("Sorry, Unknown index_type : ")

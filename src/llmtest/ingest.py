from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
)
from llmtest import constants
from langchain.document_loaders.json_loader import JSONLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.document_loaders import (
    DirectoryLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)


def mountGoogleDrive(mount_location="/content/drive"):
    from google.colab import drive
    # This will prompt for authorization.
    drive.mount(mount_location)


def get_csv_docs(csv_api_docs_path, chunk_size=1000, chunk_overlap=100, loader_kwargs=None):
    if loader_kwargs is None:
        loader_kwargs = {'csv_args': constants.CSV_DOC_PARSE_ARGS,
                         "source_column": constants.CSV_DOC_EMBEDDING_SOURCE_COLUMN}

    loader = DirectoryLoader(csv_api_docs_path, recursive=True, glob="**/*.csv", loader_cls=CSVLoader,
                             loader_kwargs=loader_kwargs)

    csv_docs = loader.load()
    csv_docs_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    csv_text = csv_docs_splitter.split_documents(csv_docs)
    return csv_text


def get_csv_docs_tiktoken(csv_api_docs_path, chunk_size=1000, chunk_overlap=100, loader_kwargs=None,
                          model_name="gpt-3.5-turbo", encoding_name="cl100k_base"):
    if loader_kwargs is None:
        loader_kwargs = {'csv_args': constants.CSV_DOC_PARSE_ARGS,
                         "source_column": constants.CSV_DOC_EMBEDDING_SOURCE_COLUMN}

    loader = DirectoryLoader(csv_api_docs_path, recursive=True, glob="**/*.csv", loader_cls=CSVLoader,
                             loader_kwargs=loader_kwargs)

    csv_docs = loader.load()
    csv_docs_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size,
                                                                             chunk_overlap=chunk_overlap,
                                                                             encoding_name=encoding_name,
                                                                             model_name=model_name)
    csv_text = csv_docs_splitter.split_documents(csv_docs)
    return csv_text


def get_json_docs(json_api_docs_path, chunk_size=1000, chunk_overlap=100, loader_kwargs=None):
    if loader_kwargs is None:
        loader_kwargs = {'jq_schema': '.', 'text_content': False}
    loader = DirectoryLoader(json_api_docs_path, recursive=True, glob="**/*.json", show_progress=True,
                             loader_cls=JSONLoader, loader_kwargs=loader_kwargs)
    json_docs = loader.load()
    json_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size,
                                                                              chunk_overlap=chunk_overlap)
    json_text = json_text_splitter.split_documents(json_docs)
    return json_text


def getHTMLDocs(html_api_docs_path, chunk_size=1000, chunk_overlap=100):
    loader = DirectoryLoader(html_api_docs_path, recursive=True, glob="**/*.html", loader_cls=UnstructuredHTMLLoader)
    html_docs = loader.load()
    html_text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=chunk_size,
                                                                      chunk_overlap=chunk_overlap)
    html_text = html_text_splitter.split_documents(html_docs)
    return html_text


def getMarkDownDocs(md_docs_path, chunk_size=1000, chunk_overlap=100):
    loader = DirectoryLoader(md_docs_path, recursive=True, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    md_docs = loader.load()
    md_text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=chunk_size,
                                                                    chunk_overlap=chunk_overlap)
    md_text = md_text_splitter.split_documents(md_docs)
    return md_text

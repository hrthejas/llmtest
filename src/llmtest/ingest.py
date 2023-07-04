from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
)

from langchain.document_loaders.json_loader import JSONLoader

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


def mountGoogleDrive(mount_location="/content/drive"):
    from google.colab import drive
    # This will prompt for authorization.
    drive.mount(mount_location)


def getJSONDocs(json_api_docs_path, chunk_size=1000, chunk_overlap=100):
    loader = DirectoryLoader(json_api_docs_path, recursive=True, glob="**/*.json", loader_cls=JSONLoader,
                             loader_kwargs={'jq_schema': '.', 'text_content': 'False'})
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

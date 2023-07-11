# import torch
# import os
# from transformers import BitsAndBytesConfig
# from enum import Enum
#
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     pipeline,
#     AutoModelForQuestionAnswering,
#     GenerationConfig
# )
#
# from langchain import (
#     HuggingFaceHub,
#     HuggingFacePipeline
# )
#
# from langchain.chains import RetrievalQA
# from langchain.embeddings import (
#     HuggingFaceEmbeddings,
#     HuggingFaceInstructEmbeddings
# )
#
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.vectorstores import (
#     Chroma,
#     FAISS,
#     ElasticVectorSearch
# )
#
# from langchain.text_splitter import (
#     CharacterTextSplitter,
#     RecursiveCharacterTextSplitter,
#     Language
# )
# from langchain.document_loaders import (
#     DirectoryLoader,
#     CSVLoader,
#     EverNoteLoader,
#     PyMuPDFLoader,
#     TextLoader,
#     UnstructuredEmailLoader,
#     UnstructuredEPubLoader,
#     UnstructuredHTMLLoader,
#     UnstructuredMarkdownLoader,
#     UnstructuredODTLoader,
#     UnstructuredPowerPointLoader,
#     UnstructuredWordDocumentLoader,
#     UnstructuredAPIFileLoader,
# )

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings
)


def get_embeddings(embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large"):
    return embedding_class(model_name=model_name)


def get_openai_embeddings():
    return OpenAIEmbeddings()

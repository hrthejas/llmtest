from enum import Enum


class IndexType(Enum):
    FAISS_INDEX = "faiss"
    CHROMA_INDEX = "chroma"
    ELASTIC_SEARCH_INDEX = "elastic"

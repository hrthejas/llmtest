from enum import Enum


class IndexType(Enum):
    FAISS_INDEX = 1
    CHROMA_INDEX = 2
    ELASTIC_SEARCH_INDEX = 3

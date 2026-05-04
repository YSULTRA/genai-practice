from langchain_chroma import Chroma
from embedding import embedding_model

vectorStore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model,
    collection_name="newRAG"
)


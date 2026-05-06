from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorDatabase:
    def __init__(self, chroma_dir: str, embedding_model: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore = Chroma(
            collection_name="rag_docs",
            persist_directory=chroma_dir,
            embedding_function=self.embeddings,
        )

    def retrieve(self, query: str, k: int = 3):
        return self.vectorstore.similarity_search(query, k=k)
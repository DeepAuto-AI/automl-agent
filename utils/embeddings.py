from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def chunk_and_retrieve(
    ref_text: str,
    documents: list,
    top_k: int,
    ranker: str = "compression",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    if len(documents) > 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        if ranker == "compression":
            embeddings_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
            retriever = FAISS.from_documents(texts, embeddings_model).as_retriever(
                search_kwargs={
                    "k": top_k * 10
                }  # chunk size is 10 times larger than the actual number of documents (i.e., we expect to have ~10% of content / document)
            )

            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            compressor = CrossEncoderReranker(model=model, top_n=top_k * 10)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            return compression_retriever.get_relevant_documents(ref_text)
        elif ranker == "bm25":
            retriever = BM25Retriever.from_documents(texts, k=top_k * 10)
            return retriever.invoke(ref_text)
    else:
        return []

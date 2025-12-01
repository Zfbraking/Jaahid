import os
from langchain_community.vectorstores import Chroma
import shutil

def build_vectorstore(chunks, embedding_model):
    # Build a fresh in-memory Chroma store every time
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )
    return vectordb


def get_retriever(vectordb, k=5):
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

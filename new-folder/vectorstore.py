import traceback
from langchain_community.vectorstores import Chroma


def build_vectorstore(chunks, embedding_model):
    """
    Build a fresh in-memory Chroma store every time from text chunks + embedding model.
    """
    try:
        print(f"[DEBUG] build_vectorstore: {len(chunks)} chunks")
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
        )
        print("[DEBUG] build_vectorstore: Chroma index created")
        return vectordb
    except Exception as e:
        print("[ERROR] build_vectorstore failed:", e)
        traceback.print_exc()
        # Re-raise so /register_pdf can return a 500 with a useful message
        raise


def get_retriever(vectordb, k=5):
    print("[DEBUG] get_retriever: k =", k)
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile, os

def extract_pdf_text(upload_file) -> str:
    try:
        if not upload_file:
            return ""
        if isinstance(upload_file, str):
            return (extract_text(upload_file) or "").strip()
        if hasattr(upload_file, "save"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                upload_file.save(tmp.name)
                text = extract_text(tmp.name) or ""
            os.remove(tmp.name)
            return text.strip()
    except Exception as e:
        print("[ERROR] PDF extraction failed:", e)
        return "PDF extraction failed"
    return ""



def chunk_text(raw_text: str, chunk_size=1000, overlap=200):
    if not raw_text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(raw_text)


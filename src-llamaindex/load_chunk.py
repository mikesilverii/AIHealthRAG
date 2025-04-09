from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
import os
import fitz as pymu
import re


def load_documents(path):
    documents = SimpleDirectoryReader(input_dir = "./data", recursive = True).load_data()

    return documents

# def define_chunker():
#     splitter = SentenceSplitter(chunk_size = 512, chunk_overlap = 64)

#     #or use rule based, section aware chunker
#     return splitter

def extract_and_clean_text_from_pdf(pdf_path):
    """
    Extract raw text from all pages of a PDF and clean formatting.
    """
    doc = pymu.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    # Normalize whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    return full_text.strip()


def load_cleaned_pdf(pdf_path):

    text = extract_and_clean_text_from_pdf(pdf_path)
    return Document(text=text, metadata={"source": os.path.basename(pdf_path)})



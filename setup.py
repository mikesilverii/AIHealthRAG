from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os

#Setup LLM (GPT-4 or Claude etc.)

#print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))


llm = OpenAI(api_key = os.environ['OPENAI_API_KEY'], model="gpt-4.1-mini", temperature=0.0)

#Use HuggingFace embeddings (BioClinicalBERT or similar)
#embedding_model = HuggingFaceEmbedding(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
embedding_model = embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en",
    #model_name="NovaSearch/stella_en_1.5B_v5",
    #model_name="michiyasunaga/BioLinkBERT-base",
    embed_batch_size=32,  # You can tweak this based on your GPU/CPU
    #query_instruction="Represent this sentence for searching relevant passages:"
    query_instruction="Represent this medical or biomedical query to retrieve related documents."
)



splitter = SentenceSplitter(chunk_size = 256, chunk_overlap = 64)

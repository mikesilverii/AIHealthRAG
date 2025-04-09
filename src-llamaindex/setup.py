from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

#Use HuggingFace embeddings (BioClinicalBERT or similar)
embedding_model = HuggingFaceEmbedding(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

splitter = SentenceSplitter(chunk_size = 512, chunk_overlap = 64)

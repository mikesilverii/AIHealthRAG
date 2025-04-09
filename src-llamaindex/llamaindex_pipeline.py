#LlamaIndex pipeline using BioBERT as a custom chunk reranker + LLM for generation

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
#from llama_index.core.rerankers import BaseReranker

#from llama_index.llms import OpenAI
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scorer import CrossEncoderScorer  # from your earlier module
from load_chunk import load_cleaned_pdf
from setup import llm, embedding_model, splitter
from prompt import prompt, query_2
import os
from typing import List, Tuple, Any
import torch
from pydantic import PrivateAttr


# class BioBERTCrossEncoderReranker():
#     def __init__(self, scorer):
#         self.scorer = scorer

#     def rerank(self, query, nodes):
#         texts = [node.get_content() for node in nodes]
#         scores = self.scorer.score(query, texts)
#         # Attach scores to nodes and sort
#         reranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
#         return [node for node, _ in reranked]

class BioBERTCrossEncoderReranker(BaseNodePostprocessor):
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=None):
        super().__init__()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def predict_scores(self, query: str, docs: List[str]) -> List[float]:
        pairs = [(query, doc) for doc in docs]
        inputs = self._tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

            # For MNLI, logits are: [entailment, neutral, contradiction]
            probs = torch.nn.functional.softmax(logits, dim=1)
            entailment_scores = probs[:, 0]  # Index 0 = entailment score
            return entailment_scores.tolist()

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
        **kwargs) -> List[NodeWithScore]:

        query = query_bundle.query_str
        node_texts = [n.node.get_content() for n in nodes]
        scores = self.predict_scores(query, node_texts)

        # Sort nodes by score (descending)
        reranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        return [node for node, _ in reranked]
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        query = query_bundle.query_str
        node_texts = [n.node.get_content() for n in nodes]
        scores = self.predict_scores(query, node_texts)

        reranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        return [node for node, _ in reranked]



if __name__ == "__main__":
    # Load PDF docs (assumes text-extracted PDFs stored as .txt)
    pdf_path = "/Users/michaelsilverii/projects2025/AIHealthRAG/src-llamaindex/input_docs/SQ Cleft_Redacted.pdf"
    documents = [load_cleaned_pdf(pdf_path)]

    # Set up service context
    # service_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embedding_model,
    #     node_parser=splitter
    # )

    # # Build index
    # index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    Settings.llm = llm
    Settings.embded_model = embedding_model
    
    index = VectorStoreIndex.from_documents(documents)
    # Plug in custom reranker
    scorer = CrossEncoderScorer()
    reranker = BioBERTCrossEncoderReranker()
    retriever = index.as_retriever(similarity_top_k=15)

    # Build query engine with reranker
    engine = RetrieverQueryEngine.from_args(retriever=retriever,
                                            llm=llm,
                                            node_postprocessors=[reranker])

    # Ask a question
    query = query_2
    response = engine.query(query)
    print("\n\n--- Answer ---\n")
    print(response.response)

    #see ranked nodes
    query_bundle = QueryBundle(query_str=query_2)
    retrieved_nodes = retriever.retrieve(query_bundle)
    reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    for i, node in enumerate(reranked_nodes):
        print(f"\n🏅 Reranked {i+1} | BioBERT Score: {node.score:.4f}")
        print(f"Chunk:\n{node.node.get_content()[:500]}...")

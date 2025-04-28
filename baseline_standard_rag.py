from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle

from load_chunk import load_cleaned_pdf
from setup import llm, embedding_model, splitter
from prompt import query_2_v2

if __name__ == "__main__":
    # Load PDF document
    #pdf_path = "/Users/michaelsilverii/projects2025/AIHealthRAG/src-llamaindex/input_docs/SQ Cleft_Redacted.pdf"
    pdf_path = "/Users/michaelsilverii/projects2025/AIHealthRAG/src-llamaindex/input_docs/EP Mandible fx_Redacted.pdf"
    documents = [load_cleaned_pdf(pdf_path)]

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embedding_model

    # Build index
    index = VectorStoreIndex.from_documents(documents)

    # Get retriever from index
    retriever = index.as_retriever(similarity_top_k=100)  # you can tune k

    # Standard RAG query engine: retriever + LLM, no reranker
    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm
    )

    # Ask a question
    query = query_2_v2
    print("Asking question...")
    response = engine.query(query)

    print("\n\n--- Answer ---\n")
    print(response.response)

    # Optionally print retrieved chunks
    print("\n\n--- Retrieved Chunks ---\n")
    query_bundle = QueryBundle(query_str=query_2_v2)
    retrieved_nodes = retriever.retrieve(query_bundle)

    # for i, node in enumerate(retrieved_nodes):
    #     print(f"\n🔹 Top {i+1} | Score: {node.score:.4f}")
    #     print(f"Chunk:\n{node.node.get_content()[:500]}...")

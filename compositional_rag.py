import asyncio
from llama_index.core import VectorStoreIndex, Settings
from load_chunk import load_cleaned_pdf
from setup import llm, embedding_model, splitter
from decomposition import decompose_query, all_subqueries, section_titles
from llama_index.core.query_engine import RetrieverQueryEngine
from ocr_load import ocr_pdf_full
from llamaparse import llama_parse

# --- Setup ---
pdf_path = ""
index = llama_parse(pdf_path)
retriever = index.as_retriever(similarity_top_k=25)

# --- Decompose main query ---
#main_query = "Summarize this patient’s medical history."
subqueries = all_subqueries

# --- Async subquery engine ---
async def run_subquery(subquery, retriever, llm, title=None):
    engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
    print(f"→ Running: {subquery}")
    response = await engine.aquery(subquery)
    return title or subquery, response.response

async def run_all_subqueries(subqueries, retriever, llm):
    tasks = [run_subquery(sq, retriever, llm, f"Section {i+1}") for i, sq in enumerate(subqueries)]
    return await asyncio.gather(*tasks)

# --- Run the full pipeline ---
if __name__ == "__main__":
    results = asyncio.run(run_all_subqueries(subqueries, retriever, llm))

    # print("\n\n--- Final Structured Summary ---\n")
    # for i, (title, answer) in enumerate(results):
    #     print(f"### {subqueries[i]}\n{answer}\n")
    final_output = ""

    for title, (_, content) in zip(section_titles, results):
        final_output += f"[{title}:]\n{content.strip()}\n\n"

    print("\n\n--- Final Structured Clinical Summary ---\n")
    print(final_output)

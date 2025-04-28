# pip install llama-cloud-services llama-index-core
from llama_cloud_services import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

parser = LlamaParse(
    api_key="llx-GIW93pTHEl7ceYnx3KISxx78u9xadDX7qPft1sKt8x7STASu",   # or export LLAMA_CLOUD_API_KEY
    result_type="markdown",        # "text" or "json" also work
    preset="balanced",             # fast | balanced | premium | custom
    num_workers=4,                 # batch parallelism
    verbose=True,
)

def llama_parse(path):

    documents = parser.load_data("/Users/michaelsilverii/projects2025/AIHealthRAG/src-llamaindex/input_docs/EM Nevus_Redacted.pdf")  # ← not .parse()

    # Optional: additional chunking if you want ~1 k-token shards
    nodes    = SentenceSplitter(chunk_size=512, chunk_overlap=128).get_nodes_from_documents(documents)

    index    = VectorStoreIndex(nodes=nodes)

    return index

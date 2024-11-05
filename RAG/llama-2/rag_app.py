# Llama2 rag application

import torch, intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_index.core import (Settings,
        SimpleDirectoryReader,
        VectorStoreIndex
)

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model

Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

# articles available
documents = SimpleDirectoryReader("articles").load_data()

# store docs into vector DB
index = VectorStoreIndex.from_documents(documents)

# set number of docs to retreive
top_k = 1

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# query documents
query = input("Enter your query: ")
response = query_engine.query(query)


# reformat response
context = "Context:\n"
for i in range(top_k):
     context = context + response.source_nodes[i].text + "\n\n"

print(context)

# Specify your Hugging Face model repository
hf_model = "NousResearch/Llama-2-7b-hf"

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model)

# Move the model to the desired device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set the model to evaluation mode
model.eval()

# Enhanced prompt template with context emphasis and instruction
prompt_template_w_context = lambda context, query: f"""
You are an expert assistant. Below is some helpful context. Please use this information to answer the query accurately.

Context:
{context}

---

Based on this context, please answer the following question in a precise and clear manner.

Question:
{query}
"""

prompt = prompt_template_w_context(context, query)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=280)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


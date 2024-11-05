# Llama2 RAG (Retrieval-Augmented Generation) Application

This project is a Retrieval-Augmented Generation (RAG) application built with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) and [LlamaIndex](https://www.llamaindex.ai/), designed to retrieve contextually relevant documents from a set of articles and then generate precise responses based on the retrieved information. The application uses Llama 2 as the language model, combined with a custom vector store for efficient document retrieval.

## Requirements

- Python 3.8 or later
- [Pytorch](https://pytorch.org/get-started/locally/)
- [intel_extension_for_pytorch](https://github.com/intel/intel-extension-for-pytorch)
- [transformers](https://huggingface.co/docs/transformers/installation)
- [llama_index](https://pypi.org/project/llama-index/)

Install dependencies with:
```bash
pip install torch intel_extension_for_pytorch transformers llama_index
```

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mdjannatulnayem/Events-iCGB.git
   cd Events-iCGB/RAG/llama-2/rag_app.py
   ```

2. **Prepare the Article Directory**

   Create an `articles` directory in the project root and add the articles you want to use for context. Each file should contain one document to be used as a data source for retrieval.

3. **Configure the Hugging Face Model**

   Specify the Hugging Face model for the RAG model in the code (default is `"NousResearch/Llama-2-7b-hf"`). You can substitute other Hugging Face models compatible with Llama 2.

## Code Explanation

### Document Loading and Indexing

This application reads documents from the `articles` directory, chunks them, and creates a vectorized index for efficient retrieval. Here’s a brief explanation of the key components:

- **Settings and Embedding Model**: This code uses `BAAI/bge-small-en-v1.5` as the embedding model, but you can change it to an alternative like `"thenlper/gte-large"`.
  
- **Vector Store Index**: All documents are converted into embeddings and stored in a vectorized index for retrieval.

### Query Processing

1. The user’s query is processed using `VectorIndexRetriever`, which searches for documents in the vector store most relevant to the query.
2. The `SimilarityPostprocessor` filters out results with low relevance based on a cutoff similarity score (`0.5` by default).
3. The relevant context is extracted from the retrieved document(s) and included in the final prompt.

### Response Generation

The retrieved context and user query are combined to create a prompt, which is fed into the Llama 2 model. The model generates an answer, using the context as reference material to enhance accuracy.

## Running the Application

Run the main script with:
```bash
python main.py
```

The application will:

1. Prompt you to enter a query.
2. Retrieve relevant context from the indexed documents.
3. Generate a response based on the context.

### Example

```plaintext
Enter your query: What are the applications of AI in healthcare?
```

Example response (assuming context from relevant articles is available):
```plaintext
AI in healthcare is applied across diagnostics, personalized medicine, and predictive analytics, enabling early disease detection and treatment planning.
```

## Customization

- **Embedding Model**: Change the embedding model by updating `Settings.embed_model` with another model name.
- **Top K Results**: Adjust `top_k` to retrieve more documents.
- **Similarity Cutoff**: Modify `SimilarityPostprocessor` to control how strict or lenient retrieval relevance should be.

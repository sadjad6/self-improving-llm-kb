# Retrieval Augmented Generation (RAG)

#rag #retrieval #llm

## What is RAG?

Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant information from an external knowledge base before generating an answer. This grounds the model's output in factual data and reduces hallucination.

## RAG Pipeline

### 1. Indexing Phase (Offline)
- **Document ingestion**: Load and parse documents
- **Chunking**: Split documents into manageable pieces
- **Embedding**: Convert chunks into dense vectors
- **Storage**: Store vectors in a vector database (e.g., FAISS, Pinecone)

### 2. Retrieval Phase (Online)
- **Query embedding**: Convert user query to a vector
- **Similarity search**: Find top-k most similar chunks
- **Retrieval strategies**:
  - **Dense retrieval**: Semantic similarity via embeddings
  - **Sparse retrieval**: Keyword matching (BM25)
  - **Hybrid retrieval**: Combination of dense + sparse

### 3. Generation Phase
- **Context construction**: Format retrieved chunks as context
- **Prompt engineering**: Create a prompt with instructions + context + query
- **LLM generation**: Generate answer grounded in retrieved context

## Chunking Strategies

### Fixed-Size Chunking
- Split text into chunks of N tokens
- Simple but may break semantic boundaries
- Overlapping windows help preserve context

### Semantic Chunking
- Split at natural boundaries (paragraphs, sections)
- Preserves meaning and context
- More complex to implement but higher quality

### Hierarchical Chunking
- Maintain document structure (headings, sections)
- Enable multi-level retrieval
- Useful for structured documents like [[Obsidian]] notes

## Retrieval Quality

### Dense Retrieval
- Uses embedding models (e.g., all-MiniLM-L6-v2, text-embedding-3-small)
- Captures semantic meaning
- Good for paraphrased queries

### Sparse Retrieval (BM25)
- Term-frequency based scoring
- Excels at exact keyword matching
- Computationally efficient

### Hybrid Approaches
- Combine dense + sparse scores with configurable weights
- Reciprocal Rank Fusion (RRF) is a popular combination method
- Typically outperforms either method alone

### Reranking
- Cross-encoder models score query-document pairs jointly
- More accurate but slower than bi-encoder retrieval
- Used as a second stage after initial retrieval

## Evaluation

Key metrics for RAG systems:
- **Retrieval**: Recall@K, MRR, nDCG
- **Generation**: Faithfulness, relevance, completeness
- **End-to-end**: Answer accuracy, hallucination rate

## Related Topics

- [[Transformers and Attention]]
- [[Machine Learning Basics]]
- [[Vector Databases]]
- [[Prompt Engineering]]


# RAG Agent

A Retrieval-Augmented Generation (RAG) system with built-in evaluation framework. Load documents, build a vector store, and query with LLM-powered responses.

## Features

- **Document Loading** - PDF, TXT, CSV, Excel, Word, JSON support
- **Vector Store** - FAISS-based semantic search with sentence transformers
- **RAG Search** - Query documents with Groq LLM-powered responses
- **Evaluation Framework** - Comprehensive retrieval quality metrics

## Quick Start

```bash
# Clone and setup
git clone https://github.com/vedsub/RAGAgent.git
cd RAGAgent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
echo "GROQ_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic RAG Query

```bash
python app.py
```

This loads documents from `data/`, builds the FAISS index, and runs a sample query.

### Evaluation Framework

Run retrieval quality evaluations on your RAG system:

```bash
# Generate synthetic dataset + run evaluation
python run_eval.py --generate-dataset --num-samples 50 --run-eval --k 3 5 10

# Run evaluation only (faster, no LLM judge)
python run_eval.py --run-eval --no-judge --k 1 3 5

# See all options
python run_eval.py --help
```

**Metrics computed:**
- Precision@K, Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- LLM-as-Judge context relevance (1-5 scale)

## Project Structure

```
├── app.py                 # Main entry point
├── run_eval.py            # Evaluation CLI
├── src/
│   ├── data_loader.py     # Multi-format document loading
│   ├── embedding.py       # Sentence transformer embeddings
│   ├── vectorstore.py     # FAISS vector store
│   ├── search.py          # RAG search with Groq LLM
│   ├── eval_dataset.py    # Synthetic dataset generator
│   ├── eval_metrics.py    # Retrieval metrics
│   ├── eval_judge.py      # LLM-as-judge scoring
│   └── eval_runner.py     # Evaluation orchestrator
├── data/                  # Your documents (PDF, TXT, etc.)
├── faiss_store/           # Persisted vector index
└── eval_data/             # Evaluation datasets & reports
```

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `GROQ_API_KEY` | Required. Get from [Groq Console](https://console.groq.com) |

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## License

MIT

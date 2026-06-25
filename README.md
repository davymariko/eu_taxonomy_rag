# EU Taxonomy RAG

A minimal, production-quality **Retrieval-Augmented Generation (RAG)** application that answers questions about the EU Taxonomy using only an official FAQ document as its knowledge source.

---

## Setup — Local

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone and install

```bash
git clone <repo-url>
cd eu-taxonomy-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
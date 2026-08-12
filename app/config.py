import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; RAW_DIR=DATA/'raw'; INDEX_DIR=DATA/'index'; MODEL_DIR=ROOT/'models'

# --- Render free tier has no GPU and ~512MB RAM, so the cloud build uses much smaller local
# models than the local/desktop version (which used bge-m3 + bge-reranker-v2-m3 on CUDA). These
# are still real embedding/reranking models, not stubs — just sized to fit the free-tier ceiling.
DEVICE=os.getenv('DEVICE','cpu')
EMBED_MODEL=os.getenv('EMBED_MODEL',str(MODEL_DIR/'bge-small-en-v1.5'))
EMBED_MODEL_HF=os.getenv('EMBED_MODEL_HF','BAAI/bge-small-en-v1.5')  # used only by download_models.py

# Reranking backend: 'local' uses a small CPU cross-encoder (default, best quality/latency).
# 'groq' asks the Groq LLM to score/reorder candidates instead of loading a second local model —
# use this only if you hit Render's free-tier memory ceiling with 'local' (see README-CLOUD.md).
RERANK_BACKEND=os.getenv('RERANK_BACKEND','local')
RERANKER_MODEL=os.getenv('RERANKER_MODEL',str(MODEL_DIR/'ms-marco-MiniLM-L-6-v2'))
RERANKER_MODEL_HF=os.getenv('RERANKER_MODEL_HF','cross-encoder/ms-marco-MiniLM-L-6-v2')
RERANKER_DEVICE=os.getenv('RERANKER_DEVICE','cpu')

# --- Groq (OpenAI-compatible) for generation, replacing local Ollama ---
GROQ_API_KEY=os.getenv('GROQ_API_KEY','')
GROQ_URL=os.getenv('GROQ_URL','https://api.groq.com/openai/v1/chat/completions')
# Groq's free-tier model lineup changes fairly often — check console.groq.com/docs/models
# before relying on this default in production.
GROQ_MODEL=os.getenv('GROQ_MODEL','openai/gpt-oss-20b')

TOP_K_SEM=int(os.getenv('TOP_K_SEM','30')); TOP_K_BM25=int(os.getenv('TOP_K_BM25','30'))
RERANK_CANDIDATES=int(os.getenv('RERANK_CANDIDATES','20'))  # smaller than the local build to keep CPU rerank latency reasonable on Render's 0.1 CPU
TOP_K_FINAL=int(os.getenv('TOP_K_FINAL','8'))
RRF_K=int(os.getenv('RRF_K','60'))
MAX_CONTEXT_CHARS=int(os.getenv('MAX_CONTEXT_CHARS','18000'))  # smaller than local build: keeps Groq prompt/latency/token-cost down

os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('HF_DATASETS_OFFLINE','1'); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')  # avoid CPU thread-pool overhead fighting Render's 0.1 CPU allocation

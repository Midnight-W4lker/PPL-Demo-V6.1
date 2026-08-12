"""
Run this BEFORE `python -m app.ingest` locally, and again as Render's build command.

It downloads the small embedding + reranker models to ./models/ so the running app can load
them with local_files_only=True (no network calls, no HF rate limits, fast cold starts on
Render's free tier). Models are intentionally small (see app/config.py) to fit ~512MB RAM.
"""
import os
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import MODEL_DIR, EMBED_MODEL_HF, RERANKER_MODEL_HF

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embed_path = MODEL_DIR / 'bge-small-en-v1.5'
    rerank_path = MODEL_DIR / 'ms-marco-MiniLM-L-6-v2'

    if not embed_path.exists():
        print(f'Downloading embedding model {EMBED_MODEL_HF} -> {embed_path}')
        SentenceTransformer(EMBED_MODEL_HF, device='cpu').save(str(embed_path))
    else:
        print(f'Embedding model already present at {embed_path}')

    if not rerank_path.exists():
        print(f'Downloading reranker model {RERANKER_MODEL_HF} -> {rerank_path}')
        CrossEncoder(RERANKER_MODEL_HF, device='cpu').save(str(rerank_path))
    else:
        print(f'Reranker model already present at {rerank_path}')

    print('Model download complete.')

if __name__ == '__main__':
    main()

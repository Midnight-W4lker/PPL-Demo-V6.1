# PPL Enterprise Intelligence — Cloud (Render free tier + Groq)

This is the same document-grounded RAG pipeline as the local/desktop version — hybrid BM25 +
dense retrieval, RRF fusion, cross-encoder reranking, confidence-hedged evidence, forced
citations — re-packaged to run on Render's free web-service tier with Groq for generation
instead of local Ollama.

**No capability was removed.** What changed is *where* each stage runs and *how big* the local
models are, because Render's free tier gives you ~512MB RAM, 0.1 shared CPU, no GPU, and no
persistent disk:

| Stage | Local/desktop version | Cloud version |
|---|---|---|
| Encoding/embedding | BAAI/bge-m3 (568M params) on CUDA | BAAI/bge-small-en-v1.5 (33M params) on CPU |
| Reranking | BAAI/bge-reranker-v2-m3 (568M) on CPU | cross-encoder/ms-marco-MiniLM-L-6-v2 (~23M) on CPU, or Groq-based reranking (`RERANK_BACKEND=groq`) if you hit the RAM ceiling |
| Generation | Local Ollama (Qwen 2.5 7B) | Groq API (default `openai/gpt-oss-20b`, swappable) |
| Retrieval (BM25 + FAISS + RRF) | unchanged | unchanged |
| Table/label extraction, confidence hedging, categorization | unchanged | unchanged |

The smaller embedding/reranker models are a real, necessary trade-off for the free tier's RAM
ceiling, not a fake substitute — they're still genuine semantic models, just sized to fit. If
retrieval quality matters more to you than free hosting, run this on Render's **Standard** tier
($25/mo, 2GB RAM) instead and swap `EMBED_MODEL_HF`/`RERANKER_MODEL_HF` back to the bge-m3 /
bge-reranker-v2-m3 used locally.

## Why the repo layout looks different from the local version

Render's free tier has **no persistent disk** — anything written at runtime disappears on
restart. So unlike the local version (which `.gitignore`s `data/raw/`, `data/index/`, and
`models/`), this cloud repo **intentionally commits**:
- `data/raw/*.pdf` — needed at request time by the `/api/pdf/page/{page}` endpoint (renders page
  images for citation click-through).
- `data/index/*` — the prebuilt FAISS/BM25/records index, so the service never needs to
  re-ingest on boot.

`models/` is still gitignored — the small embedding/reranker weights are fetched at **build
time** by `app/download_models.py` instead, so you're not committing ~150-200MB of binary
weights to git.

## One-time local setup (before your first deploy)

```bash
pip install -r requirements.txt
python -m app.download_models     # fetches the small models into models/
python -m app.ingest               # builds data/index/ using those same small models
```

Run this locally whenever you add a new report to `data/raw/` — it must run on your machine
(or in CI) before you push, since Render's free build environment shouldn't be relied on to hold
the full ingestion pipeline's peak memory reliably either.

## Deploy to Render

The Docker build now does everything — downloads the small models **and** builds the FAISS/BM25
index from whatever PDFs are in `data/raw/`, at build time. You don't need a working local Python
environment just to deploy; you only need one if you want to test locally first (recommended) or
run the eval harness.

1. Push this folder to a GitHub repo — `Dockerfile`, `requirements.txt`, `app/`, `data/raw/*.pdf`
   all at the repo root (not nested in a subfolder, or Render's auto-detect/Docker context won't
   find them).
2. In Render: **New → Blueprint**, point it at the repo — `render.yaml` is already set up.
   (Or **New → Web Service** and let it auto-detect the Dockerfile.)
3. In the Render dashboard, set the `GROQ_API_KEY` secret (get one free at
   [console.groq.com](https://console.groq.com)). Never commit it.
4. Deploy. First build takes several minutes (downloading models + extracting/embedding the PDF)
   — that's expected and one-time per deploy. After that, Render's free tier still sleeps after
   15 minutes of inactivity and cold-starts on the next request; that's a platform limit on the
   free plan, not something the app can avoid.
5. Once it's up, hit `/api/health` first — it reports `{"index": true/false, "groq": true/false}`
   so you can immediately tell whether the index built correctly and whether the Groq key is set,
   without needing to dig through logs.

Adding a new report later: drop the PDF into `data/raw/`, push, and Render rebuilds the image
(and therefore the index) automatically. No local ingest step required.

## Optional: test locally before deploying

I could not execute the model download or a live Groq call in my own sandbox (its network
allowlist doesn't include huggingface.co or api.groq.com), so please run this yourself first:

```bash
python -m app.download_models
python -m app.ingest
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --reload
# then in another terminal:
curl -s localhost:8000/api/health
curl -s -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"query":"What was PPL profit after tax as a percentage of net sales in FY 2024-25?"}'
```

If that returns a cited, hedged-where-appropriate answer, you're good to push. Also run
`python -m tests.run_eval` against the local server for the golden-question tripwire.

## If you hit the 512MB RAM ceiling

Render's free instance can genuinely OOM with two loaded models (embedder + reranker) plus
PyTorch/FastAPI overhead. In order of preference:

1. Set `RERANK_BACKEND=groq` (env var) — drops the local cross-encoder entirely and reranks via
   a Groq prompt instead. Slightly higher latency, no local RAM cost.
2. Swap `EMBED_MODEL_HF` to `sentence-transformers/all-MiniLM-L6-v2` (22M params, smaller than
   bge-small) if the embedder alone is too heavy — re-run `download_models.py` + `ingest.py`
   after changing it, since the query-time embedding model must match the one used to build the
   index.
3. If it still OOMs, the free tier's 512MB is simply too small for this combination — move to
   Render's Standard plan ($25/mo, 2GB RAM), which comfortably fits even the original bge-m3 +
   bge-reranker-v2-m3 models on CPU.

## Env vars reference

| Var | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | set as a Render secret, never commit |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq's free-tier model lineup changes often — check console.groq.com/docs/models |
| `RERANK_BACKEND` | `local` | `local` or `groq` |
| `EMBED_MODEL_HF` / `RERANKER_MODEL_HF` | bge-small-en-v1.5 / ms-marco-MiniLM-L-6-v2 | only read by `download_models.py` |

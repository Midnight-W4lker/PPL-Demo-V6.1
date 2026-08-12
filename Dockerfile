FROM python:3.11-slim

WORKDIR /app

# PyMuPDF/faiss-cpu ship prebuilt wheels for this base image; no extra system build deps needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the small embedding/reranker models at BUILD time (Render's build machine has internet
# access, unlike the running free-tier container's constrained environment) so they're baked
# into the image and load with local_files_only=True at request time — no network dependency,
# no cold-start download.
RUN python -m app.download_models

# Build the FAISS/BM25 index from whatever PDFs are already committed under data/raw/ — also at
# build time, using the same models just downloaded. This means the image is self-contained and
# deployable with zero manual local steps; there's no "did you remember to run ingest.py first"
# failure mode. Re-running `docker build` after adding a new PDF to data/raw/ rebuilds the index
# automatically.
RUN python -m app.ingest

# Render injects $PORT at runtime; shell form so it's expanded, not treated literally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]

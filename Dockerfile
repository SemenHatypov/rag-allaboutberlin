# Streamlit RAG app image. The same image runs on Hugging Face Spaces
# (Docker SDK, port 7860) and on any plain Docker host (see docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# Hugging Face Spaces runs containers as UID 1000; keep the venv, the model
# cache, and the app itself under that user's home so everything stays readable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /home/user/app

COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Bake both MiniLM models into the image so cold starts never hit the network.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"
# Serve models from the baked cache only — no HF Hub calls at runtime.
ENV HF_HUB_OFFLINE=1

COPY --chown=user:user app.py ingest.py rag_helper.py ./
COPY --chown=user:user .streamlit ./.streamlit
COPY --chown=user:user output/json ./output/json

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '7860') + '/_stcore/health')"

# Shell form is needed for ${PORT} expansion; exec makes streamlit PID 1.
CMD exec streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0

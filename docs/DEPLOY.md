# Deployment

The app is a stateless Streamlit container: corpus data (`output/json/`) and both
MiniLM models are baked into the image, the only runtime input is `OPENAI_API_KEY`.
The same [Dockerfile](../Dockerfile) serves every target below.

## Hugging Face Space (primary, always available)

The Space auto-wakes on any visitor — no owner action needed, unlike Streamlit
Community Cloud. Free tier: 2 vCPU / 16 GB RAM, sleeps only after 48 h without
traffic and restarts itself in ~30–60 s on the next visit.

The Space runs the Docker image built from the repo [Dockerfile](../Dockerfile)
(Docker SDK — requires a **HF PRO subscription**; `sdk: docker` and `app_port`
come from the README front matter).

One-time setup:

1. On huggingface.co (with PRO active): **New Space** → SDK **Docker**,
   visibility **Public**, hardware **CPU basic**. ZeroGPU is not needed —
   the models are tiny and CPU-only. Leave the Space empty (no template) —
   the first push below overwrites it anyway.
2. In the Space: **Settings → Variables and secrets** → add secret
   `OPENAI_API_KEY`.
3. On hf.co **Settings → Access Tokens**: create a token with **write** scope.
4. In the GitHub repo: **Settings → Secrets and variables → Actions** →
   - secret `HF_TOKEN` = the token from step 3;
   - variable `HF_SPACE` = `<hf-username>/<space-name>`.
5. Run the **Deploy to Hugging Face Space** workflow manually once
   (Actions → Deploy to Hugging Face Space → Run workflow) and watch the
   Space build logs.

After that every push to `main` re-deploys automatically
([deploy-space.yml](../.github/workflows/deploy-space.yml)); the weekly
re-scrape pushes its refreshed corpus to the Space itself
([rescrape.yml](../.github/workflows/rescrape.yml)), because commits made with
`GITHUB_TOKEN` never trigger other workflows.

App URLs: `https://huggingface.co/spaces/<user>/<space>` (with HF chrome) or
the bare app at `https://<user>-<space>.hf.space` (dashes replace any `_`/`.`).

## Any Docker host (VPS, a friend's server, …)

```bash
git clone https://github.com/<you>/rag-allaboutberlin.git
cd rag-allaboutberlin
echo 'OPENAI_API_KEY=sk-...' > .env
docker compose up -d --build
```

The app listens on port 7860 (override with `PORT`). Put a reverse proxy
(Caddy, nginx, Traefik) in front for HTTPS/domains. The container exposes
`/_stcore/health` and ships a `HEALTHCHECK`, so `restart: unless-stopped`
plus the health status covers supervision.

To update: `git pull && docker compose up -d --build`.

## Streamlit Community Cloud (legacy mirror)

Still deployed at [rag-all-about-berlin.streamlit.app](https://rag-all-about-berlin.streamlit.app/)
from `main` with the `OPENAI_API_KEY` secret set in the app settings. Free-tier
apps hibernate after ~12 h without traffic and take minutes to wake, which is
why the HF Space is the link to share.

# BotHunter Deployment Guide

Step-by-step instructions for running BotHunter in every environment.

---

## 1. Local Development

```bash
# Clone and enter the project
git clone https://github.com/sudhishkg11/BotHunter.git
cd BotHunter

# Install dependencies
pip install -e "."          # core deps only
pip install -e ".[dev]"     # + test tools
pip install -e ".[ai]"      # + AI provider SDKs

# Set up environment variables (copy and edit)
cp .env.example .env

# Initialize the database and seed synthetic data
python database.py
python ingestor.py

# Launch dashboard
python -m streamlit run app.py

# Launch API server (separate terminal)
uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 2. Docker (single service)

```bash
# Build image
docker build -t bothunter .

# Run dashboard on port 8501
docker run -p 8501:8501 \
  -e AI_ENABLED=false \
  bothunter

# Run API on port 8000
docker run -p 8000:8000 \
  -e AI_ENABLED=false \
  bothunter \
  python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 3. Docker Compose (both services)

```bash
# Copy and fill in your env vars
cp .env.example .env
# Edit AI_API_KEY and DATABASE_URL as needed

# Start both dashboard (8501) and API (8000)
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

Persistent data lives in the `bothunter_data` Docker volume.

---

## 4. Render.com (free cloud hosting)

### Prerequisites
- A [Render account](https://render.com) (free tier available)
- Your GitHub repo pushed and public (or connected to Render)

### Steps
1. Go to **Render Dashboard → New → Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — click **Apply**
4. In **Environment**, set `AI_API_KEY` to your provider key (if using AI features)
5. Click **Deploy**

The app will be live at `https://bothunter.onrender.com` (or your custom domain).

> **Note**: Free tier instances spin down after 15 minutes of inactivity and take ~30s to cold-start.

---

## 5. Railway (alternative cloud)

1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Deploy:
   ```bash
   railway init
   railway up
   ```
4. Set env vars in the Railway dashboard or via:
   ```bash
   railway variables set AI_API_KEY=sk-...
   ```

---

## Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | `sqlite:///bothunter.db` | Full SQLAlchemy DB connection string |
| `AI_PROVIDER` | No | `openai` | `openai` \| `gemini` \| `anthropic` \| `ollama` |
| `AI_API_KEY` | If AI enabled | — | API key for the chosen AI provider |
| `AI_MODEL` | No | `gpt-4o-mini` | Model name (provider-specific) |
| `AI_ENABLED` | No | `false` | `true` to enable AI explanations |
| `OLLAMA_BASE_URL` | Ollama only | `http://localhost:11434` | Ollama server URL |
| `SECRET_KEY` | Production | — | Secret key for API request signing |

---

## Upgrading to PostgreSQL

Change `DATABASE_URL` to a Postgres connection string:

```
DATABASE_URL=postgresql+psycopg2://user:password@host:5432/bothunter
```

Then install the driver:

```bash
pip install psycopg2-binary
# or: pip install -e ".[deploy]"
```

All SQLAlchemy models are fully compatible — no code changes needed.

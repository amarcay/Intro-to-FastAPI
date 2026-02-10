FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY tests/ tests/
COPY workflow_demo.ipynb README.md ./

EXPOSE 8000 8501

CMD ["sh", "-c", "uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & uv run uvicorn src.app.api:app --host 0.0.0.0 --port 8000"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY orchardflow ./orchardflow
COPY review_app.py ./review_app.py

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[review]"

CMD ["python", "-m", "orchardflow.demo"]

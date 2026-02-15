# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN python -m pip install --upgrade pip

# ---- deps layer (cached) ----
# Copy requirements so it's available for later layers too (important for dynamic deps)
COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---- app layer ----
# Copy packaging metadata (required for pip install .)
COPY pyproject.toml README.md ./

# Copy source + runtime assets
COPY src ./src
COPY templates ./templates
COPY static ./static
COPY config ./config

# Install your package (don't reinstall deps since we already did requirements)
RUN pip install . --no-deps

# Make sure appuser can read app files
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "vehicle_insurance.app:app", "--host", "0.0.0.0", "--port", "8000"]

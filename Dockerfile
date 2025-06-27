# Use optimized Python + uv base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set work directory
WORKDIR /app

# Avoid apt prompts and enable faster install
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install `make` with minimal overhead
RUN apt-get update && \
    apt-get install -y --no-install-recommends make && \
    rm -rf /var/lib/apt/lists/*

# Copy lockfile and project config first for cache efficiency
COPY uv.lock pyproject.toml ./

# Install dependencies using cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy remaining project files
COPY . .

# Install project (with cache)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Pre-create DB tables
RUN make create-tables

# Add local venv to PATH
ENV PATH="/app/.venv/Lib:$PATH"

# Avoid `uv` entrypoint override
ENTRYPOINT []

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

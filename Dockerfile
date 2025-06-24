# Start with the official Python 3.12 slim-buster image.
# slim-buster is a good choice for production as it's smaller.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/Lib:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Command to run the application
# Use uvicorn to run the FastAPI application.
# --host 0.0.0.0 is required to make the app accessible from outside the container
# --port 8000 to match the exposed port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
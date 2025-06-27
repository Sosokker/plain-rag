.PHONY: install-deps create-tables help start

help:
	@echo "Available targets:"
	@echo "  create-tables  - Create database tables using create_tables.py"
	@echo "  install-deps   - Install Python dependencies using uv"
	@echo "  start          - Start fastAPI server on port 8000"
	@echo "  help           - Show this help message"

install-deps:
	uv sync --locked --no-install-project --no-dev

create-tables:
	@echo "Creating database tables..."
	uv run python scripts/create_tables.py

start:
	uv run uvicorn app.main:app --port 8000

.DEFAULT_GOAL := help
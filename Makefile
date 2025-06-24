.PHONY: create-tables help

help:
	@echo "Available targets:"
	@echo "  create-tables  - Create database tables using create_tables.py"
	@echo "  install-deps   - Install Python dependencies using uv"
	@echo "  help           - Show this help message"

install-deps:
	uv sync --locked --no-install-project --no-dev

create-tables:
	@echo "Creating database tables..."
	uv run python scripts/create_tables.py

.DEFAULT_GOAL := help
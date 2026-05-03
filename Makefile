.PHONY: install run lint test clean

install:
	uv sync

run:
	uv run streamlit run app/app.py

lint:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

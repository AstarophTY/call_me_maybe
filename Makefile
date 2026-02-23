TEMP_FILE := *.egg-info build dist .pytest_cache .mypy_cache

all: run

run: install
	un run python -m src

install:
	uv sync

clean:
	@rm -rf $(TEMP_FILE)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete

lint:
	flake8 src
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src
	mypy src --strict
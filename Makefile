TEMP_FILE := dist .pytest_cache .mypy_cache
MYPY_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

SRC_DIR := src

UV := uv


install:
	$(UV) sync

clean:
	$(RM) -rf $(TEMP_FILE)
	find . -type d \( -name "__pycache__" -o -name "*.egg-info" -o -name "build" -o -name "dist" \) -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

lint:
	$(UV) run flake8 $(SRC_DIR)/
	$(UV) run mypy $(SRC_DIR)/ $(MYPY_FLAG) 

lint-strict:
	$(UV) run flake8 $(SRC_DIR)/
	$(UV) run mypy $(SRC_DIR)/ --strict
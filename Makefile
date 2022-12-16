SRC_DIR		= src
TEST_DIR	= tests
CHECK_DIRS = $(SRC_DIR) $(TEST_DIR)
DOCS_DIR 	= docs

.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using pyenv and poetry"
	@poetry install	
	@ poetry run pre-commit install
	@poetry shell

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking Poetry lock file consistency with 'pyproject.toml': Running poetry lock --check"
	@poetry lock --check
	@echo "🚀 formatting code: black and isort"
	@poetry run black $(CHECK_DIRS)
	@poetry run isort $(CHECK_DIRS)
	@echo "🚀 Static type checking: Running mypy"
	@poetry run mypy $(CHECK_DIRS)
	@echo "🚀 Launch the linting tool"
	@poetry run pylint -j 0 $(SRC_DIR)
	@poetry run pylint -j 0 -d missing-function-docstring $(TEST_DIR)

.PHONY: test
test: ## Launch the tests
	@echo "🚀 Testing code: Running pytest"
	@poetry run pytest $(TEST_DIR)

.PHONY: clean
clean: ## Clean the repository
	rm -rf dist
	rm -rf *.egg-info

.PHONY: update
update: ## Update python dependencies
	@poetry update

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'






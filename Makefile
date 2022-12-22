SRC_DIR		 = src
TEST_DIR	 = tests
CHECK_DIRS   = $(SRC_DIR) $(TEST_DIR)
DOCS_DIR 	 = docs
PYTEST_FLAGS = -vv -n auto --cov=src

.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using pyenv and poetry"
	@poetry install	
	@poetry update	
	@poetry shell

.PHONY: update
update: ## Update python dependencies
	@poetry update

.PHONY: build
build: ## Builds a package, as a tarball and a wheel by default.
	@poetry build

.PHONY: format
format:
	@poetry run black $(CHECK_DIRS)
	@poetry run isort $(CHECK_DIRS)

.PHONY: format-check
format-check:
	@poetry run black --check $(CHECK_DIRS)
	@poetry run isort --check $(CHECK_DIRS)

.PHONY: lint
lint: ## Launch the linting tool
	@poetry run pylint -j 0 $(SRC_DIR)
	@poetry run pylint -j 0 -d missing-function-docstring -d missing-class-docstring $(TEST_DIR)

.PHONY: type-check
type-check:
	@poetry run mypy $(CHECK_DIRS)

.PHONY: test
test: ## Launch the tests
	@poetry run pytest $(PYTEST_FLAGS) --cov-fail-under=100

.PHONY: coverage
coverage: ## Computes test coverage
	@poetry run pytest $(PYTEST_FLAGS) --cov-report html:generated/reports/coverage

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking Poetry lock file consistency with 'pyproject.toml': Running poetry lock --check"
	@poetry lock --check
	@echo "🚀 checking code formatting: black and isort"
	@make format-check
	@echo "🙄 Launch the linting tool $(SRC_DIR)"
	@make lint
	@echo "😰 Static type checking: Running mypy"
	@make type-check
	@echo "🥶 Testing code: Running pytest"
	@make test

.PHONY: clean
clean: ## Clean the repository
	rm -rf dist
	rm -rf *.egg-info
	rm -rf coverage

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@poetry run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@poetry run mkdocs serve

# .PHONY: release
# release: ## Build and Publishes a package to a remote repository.
# 	@poetry publish --build

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'






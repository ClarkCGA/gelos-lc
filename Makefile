#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = gelos-lc
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Load data-path variables from .env (also read by docker compose) so targets
# like generate-app-files/upload-app-files work outside Docker. .env values
# override shell-exported variables; to override per-invocation, pass the
# variable on the make command line: make upload-app-files PROCESSED_PATH=...
-include .env
export RAW_PATH PROCESSED_PATH INTERIM_PATH EXTERNAL_PATH

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pixi install


## Install pixi env, then editable-install ../gelos so its deps come from pip
.PHONY: dev-install
dev-install: requirements
	pip install -e "../gelos[alphaearth]"




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://gelos-fm/data 
	



## Calculate dataset statistics
.PHONY: statistics
statistics:
	python src/calculate_statistics.py $(DATA_VERSION)
	
.PHONY: generation
generation:
	python -m gelos.generation
	
.PHONY: analysis
analysis:
	python -m gelos.analysis

## Run cross-experiment comparisons
.PHONY: comparison
comparison:
	python -m gelos.comparison

## Generate all gelos-app files (json, pmtiles, config.js)
.PHONY: generate-app-files
generate-app-files:
	python src/app_files_generation.py \
		--raw-data-dir $${RAW_PATH:-/app/data/raw} \
		--processed-data-dir $${PROCESSED_PATH:-/app/data/processed} \
		--interim-data-dir $${INTERIM_PATH:-/app/data/interim} \
		--data-version v0.50.1

## Upload all gelos-app files (json, pmtiles, config.js) to s3://gelos-fm/
.PHONY: upload-app-files
upload-app-files:
	python src/app_files_upload.py \
		--processed-data-dir $${PROCESSED_PATH:-/app/data/processed} \
		--data-version v0.50.1

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	@echo ">>> Pixi environment configured in pyproject.toml. Run 'make requirements' to install dependencies."
	
	@echo ">>> Activate with:\npixi shell"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

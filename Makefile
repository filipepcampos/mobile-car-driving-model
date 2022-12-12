.PHONY: clean data lint requirements sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = avg-kitti
GTSDB_DOWNLOAD_URL = https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip
PROJECT_NAME = mobile_car_driving
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Download Data from S3
get_data:
	aws s3 sync s3://$(BUCKET) data/raw --exclude "*" --include "data_object_image_2.zip" --include "data_object_label_2.zip" --no-sign-request
	unzip data/raw/data_object_image_2.zip -d data/raw/kitti
	unzip data/raw/data_object_label_2.zip -d data/raw/kitti
	curl $(GTSDB_DOWNLOAD_URL) -o data/raw/gtsdb.zip -C -
	unzip data/raw/gtsdb.zip -d data/raw/gtsdb
	rm data/raw/*.zip

## Set up python interpreter environment for development
build_env_dev:
	$(PYTHON_INTERPRETER) -m venv .venv_dev && \
	source .venv_dev/bin/activate && \
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt && \
	$(PYTHON_INTERPRETER) -m pip install -r requirements_dev.txt

## Set up python interpreter environment for production
build_env_prod:
	$(PYTHON_INTERPRETER) -m venv .venv_prod
	source .venv_prod/bin/activate
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Install pre-commit and git hooks
install-pre-commit: build_env_dev
	@source .venv_dev/bin/activate && \
	pre-commit install --install-hooks -t pre-commit -t commit-msg && \
	pre-commit autoupdate

## Uninstall pre-commit and git hooks
uninstall-pre-commit:
	@source .venv_dev/bin/activate && \
	pre-commit uninstall && \
	pre-commit uninstall --hook-type pre-push && \
	rm -rf .git/hooks/pre-commit


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

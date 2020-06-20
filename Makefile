#################################################################################
# GLOBALS                                                                       #
#################################################################################
MAKEFLAGS = -s

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = multimodal-data-abstraction
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

# variables for active network
ACTIVE =				# indicates the active network
REGNET = regnet 		# placeholder for Regnet
MULTIMODAL = multimodal # placeholder for Multimodal

#################################################################################
# MODEL COMMANDS                                                                #
#################################################################################
.PHONY: regnet
## Sets active network to regnet
regnet:
	$(eval ACTIVE=regnet)

.PHONY: multimodal
## Sets active network to multimodal
multimodal:
	$(eval ACTIVE=multimodal)

.PHONY: active
## prints active network
active:
	$(call check_defined, ACTIVE, active network)
	if [ $(ACTIVE) = $(REGNET) ]; then \
		echo "the active network is regnet"; \
	elif [ $(ACTIVE) = $(MULTIMODAL) ]; then \
		echo "the active network is multimodal"; \
	fi

.PHONY: data
## Make Dataset
data:
	$(call check_defined, ACTIVE, active network)
ifeq ($(keep),)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).data.make_dataset
else
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).data.make_dataset --keep
endif

.PHONY: clean_data data_clean
## Destroy Dataset
clean_data data_clean:
	$(RM) -r dataset/raw/*
	$(RM) -r dataset/interim/*
	$(RM) -r dataset/processed/*

.PHONY: train learn
## Train The Model
train learn:
	$(call check_defined, ACTIVE, active network)
ifeq ($(epochs),)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).models.train_model
else
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).models.train_model --epochs $(epochs)
endif

.PHONY: train_resume learn_resume
## Resume traininig The Model
train_resume learn_resume:
	$(call check_defined, ACTIVE, active network)
	$(call check_defined, epochs, epochs to run)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).models.train_model --resume --epochs $(epochs)

.PHONY: predict
## Predict
predict:
	$(call check_defined, ACTIVE, active network)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).models.predict_model -i -d

.PHONY: predict_all
## Predict from test, valid and training datasets
predict_all:
	$(call check_defined, ACTIVE, active network)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).models.predict_all

.PHONY: predict_test
## Predict from test, valid and training datasets
predict_test:
	$(call check_defined, ACTIVE, active network)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).models.predict_test

.PHONY: refine
## Predict using Iterative Refinement
refine:
	if [ $(ACTIVE) = $(REGNET) ]; then \
		$(PYTHON_INTERPRETER) -m src.regnet.models.iterative_refinement; \
	else \
		echo "unknown network specified"; \
	fi

.PHONY: graph
## Draw performance graphs
graph:
	$(call check_defined, ACTIVE, active network)
	$(PYTHON_INTERPRETER) -m src.$(ACTIVE).visualization.visualize

.PHONY: check test
## Run nose tests
check test:
	nosetests --nologcapture --nocapture

# *:
# 	if [ $(ACTIVE) = $(REGNET) ]; then \
# 		$(MAKE) $@_$(REGNET); \
# 	elif [ $(ACTIVE) = $(MULTIMODAL) ]; then \
# 		$(MAKE) $@_$(MULTIMODAL); \
# 	else \
# 		echo "unknown network specified"; \
# 	fi

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

.PHONY: requirements
## Install Python Dependencies
requirements: test_environment
	pip install -U pip setuptools wheel
	# pip install -r requirements.txt
	for req in $$(cat requirements.txt); do pip install $$req; done


.PHONY: clean
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rv {} +

.PHONY: requirements
## Lint using flake8
lint:
	pylint --rcfile=.pylintrc src

.PHONY: sync_data_to_s3
## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

.PHONY: sync_data_from_s3
## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

.PHONY: create_environment
## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

.PHONY: test_environment
## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# CHECK IF VARIABLES ARE DEFINED                                                #
#################################################################################
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
# 
# Usage:
# 	>>> $(call check_defined, MY_FLAG)
# 	>>> $(call check_defined, OUT_DIR, build directory)
# 	>>> $(call check_defined, BIN_DIR, where to put binary artifacts)
# 	>>> $(call check_defined, LIB_INCLUDE_DIR LIB_SOURCE_DIR, library path)
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
        $(error undefined variable "$1"$(if $2, ($2))$(if $(value @), \
                required by target '$@')))

# Check that a variable specified through the stem is defined and has
# a non-empty value, die with an error otherwise.
#
#   %: The name of the variable to test.
#   
check-defined-%: __check_defined_FORCE
	@:$(call check_defined, $*, target-specific)

# Since pattern rules can't be listed as prerequisites of .PHONY,
# we use the old-school and hackish FORCE workaround.
# You could go without this, but otherwise a check can be missed
# in case a file named like `check-defined-...` exists in the root 
# directory, e.g. left by an accidental `make -t` invocation.
.PHONY : __check_defined_FORCE
__check_defined_FORCE :

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

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
.PHONY: show-help
show-help:
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

# Makefile for building llama-stack images
# Usage: make [TAG=latest] [target]

# Default tag if none specified
TAG ?= latest

# Image names
LMEVAL_IMAGE = llama-stack-lmeval:$(TAG)
GARAK_IMAGE = llama-stack-garak:$(TAG)
GUARDRAILS_IMAGE = llama-stack-guardrails:$(TAG)

# Quay.io registry tags
LMEVAL_QUAY = quay.io/ruimvieira/llama-stack-lmeval:$(TAG)
GARAK_QUAY = quay.io/ruimvieira/llama-stack-garak:$(TAG)
GUARDRAILS_QUAY = quay.io/ruimvieira/llama-stack-guardrails:$(TAG)

# Distribution directories
LMEVAL_DIR = trustyai-lmeval-distribution
GARAK_DIR = trustyai-garak-distribution
GUARDRAILS_DIR = trustyai-shields-fms-distribution

.PHONY: help check-venv pre-commit build-all build-lmeval build-garak build-guardrails clean

# Default target
help:
	@echo "Available targets:"
	@echo "  build-all      - Build all three images"
	@echo "  build-lmeval   - Build lmeval image"
	@echo "  build-garak    - Build garak image"
	@echo "  build-guardrails - Build guardrails image"
	@echo "  clean          - Remove all built images"
	@echo ""
	@echo "Usage: make [TAG=latest] [target]"
	@echo "Example: make TAG=v1.0.0 build-all"

# Check if we're in a virtual environment
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Error: Not in a virtual environment. Please activate one first."; \
		echo "You can create one with: python -m venv .venv && source .venv/bin/activate"; \
		exit 1; \
	fi
	@echo "✓ Virtual environment detected: $$VIRTUAL_ENV"

# Install llama-stack in editable mode
install: check-venv
	@echo "Installing llama-stack in editable mode..."
	pip install -e .
	@echo "✓ llama-stack installed successfully!"

# Run pre-commit checks
pre-commit: install
	@echo "Running pre-commit checks..."
	pre-commit run --all-files

# Build all images
build-all: pre-commit build-lmeval build-garak build-guardrails
	@echo "✓ All images built successfully!"

# Build lmeval image
build-lmeval: pre-commit
	@echo "Building $(LMEVAL_IMAGE)..."
	python $(LMEVAL_DIR)/build.py
	docker build -f $(LMEVAL_DIR)/Containerfile -t $(LMEVAL_IMAGE) .
	docker tag $(LMEVAL_IMAGE) $(LMEVAL_QUAY)
	@echo "✓ $(LMEVAL_IMAGE) and $(LMEVAL_QUAY) built successfully!"

# Build garak image
build-garak: pre-commit
	@echo "Building $(GARAK_IMAGE)..."
	python $(GARAK_DIR)/build.py
	docker build -f $(GARAK_DIR)/Containerfile -t $(GARAK_IMAGE) .
	docker tag $(GARAK_IMAGE) $(GARAK_QUAY)
	@echo "✓ $(GARAK_IMAGE) and $(GARAK_QUAY) built successfully!"

# Build guardrails image
build-guardrails: pre-commit
	@echo "Building $(GUARDRAILS_IMAGE)..."
	python $(GUARDRAILS_DIR)/build.py
	docker build -f $(GUARDRAILS_DIR)/Containerfile -t $(GUARDRAILS_IMAGE) .
	docker tag $(GUARDRAILS_IMAGE) $(GUARDRAILS_QUAY)
	@echo "✓ $(GUARDRAILS_IMAGE) and $(GUARDRAILS_QUAY) built successfully!"

# Clean up built images
clean:
	@echo "Removing built images..."
	docker rmi $(LMEVAL_IMAGE) $(GARAK_IMAGE) $(GUARDRAILS_IMAGE) $(LMEVAL_QUAY) $(GARAK_QUAY) $(GUARDRAILS_QUAY) 2>/dev/null || true
	@echo "✓ Cleanup completed!"

.PHONY: install train build run test clean

install:
	pip install -r requirements.txt

train:
	python src/train_pipeline.py

build:
	docker build -t ml-pipeline .

run:
	docker run -p 5000:5000 ml-pipeline

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make train    - Train the ML model"
	@echo "  make build    - Build Docker image"
	@echo "  make run      - Run Docker container"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up files"

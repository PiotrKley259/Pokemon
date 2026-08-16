.PHONY: install collect data train evaluate test dashboard all \
        images embed coldstart

install:
	pip install -r requirements.txt

collect:
	python -m src.data.collect

data: collect
	python -m src.data.build_dataset

train:
	python -m src.models.train --task all

evaluate:
	python -m src.models.evaluate --task a

test:
	pytest tests/ -v

dashboard:
	streamlit run app/dashboard.py

images:
	python -m src.data.images

embed:
	python -m src.features.embeddings --crop art
	python -m src.features.embeddings --crop full

coldstart:
	python -m src.models.train_coldstart --compare-crops

all: data train evaluate

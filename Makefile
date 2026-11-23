.PHONY: install test audit app

install:
	pip install -e .

test:
	pytest -q

audit:
	python -m tokentax_lab.cli audit --flores data/flores --output results/audit.csv

app:
	streamlit run app.py

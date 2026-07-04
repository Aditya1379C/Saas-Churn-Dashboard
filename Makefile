.PHONY: install seed scrape pipeline train serve report test clean

install:
	pip install -r requirements.txt

seed:
	python setup_stripe_data.py

scrape:
	python predict.py scrape

pipeline:
	python predict.py pipeline

train:
	python predict.py train

serve:
	python predict.py serve

report:
	python predict.py report

# Run full pipeline end-to-end (scrape → pipeline → train)
run:
	python predict.py scrape
	python predict.py pipeline
	python predict.py train

test:
	pytest tests/ -v

clean:
	rm -f data/customers.csv data/customers.db
	rm -f models/churn_model.pkl models/encoders.pkl models/feature_importance.json
	rm -f logs/*.log

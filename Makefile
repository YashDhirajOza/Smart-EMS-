install-simulator:
	pip install -r simulator/requirements.txt

run-simulator:
	python simulator/src/main.py

up:
	docker-compose up --build

teardown:
	docker-compose down -v

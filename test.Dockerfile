FROM python:3.11

WORKDIR /app

ADD . /app

RUN pip install poetry

RUN poetry install

CMD poetry run pytest . --log-file ./tests/logs/$(date '+%F_%H:%M:%S')


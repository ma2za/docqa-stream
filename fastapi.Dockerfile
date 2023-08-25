FROM python:3.11

WORKDIR /app

ADD . /app

RUN pip install poetry

RUN poetry install

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
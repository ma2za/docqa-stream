FROM python:3.8

#
WORKDIR /src

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

#
CMD ["uvicorn", "src.docqa_stream.server:app", "--host", "0.0.0.0", "--port", "80"]
FROM python:3.7.11-slim-buster
COPY requirements.txt /opt/training/requirements.txt
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /opt/training/requirements.txt
WORKDIR /training/
COPY . /training/
ENV PYTHONUNBUFFERED 1
ENTRYPOINT ["python"]
CMD ["__main__.py"]
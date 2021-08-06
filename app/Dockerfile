FROM python:3.8-slim-buster
COPY requirements.txt /opt/app/requirements.txt
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /opt/app/requirements.txt
WORKDIR /app/
COPY . /app/
ENV PYTHONUNBUFFERED 1
ENTRYPOINT ["python"]
CMD ["main.py"]
# Use an official Python runtime as a parent image
FROM python:3.10.12


WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=App.py

# Run flask when the container launches
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]

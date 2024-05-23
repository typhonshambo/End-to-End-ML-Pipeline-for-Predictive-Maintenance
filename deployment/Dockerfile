# File: deployment/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY notebooks/ ./notebooks/

# Ensure the directories for processed data and saved models exist
RUN mkdir -p data/processed
RUN mkdir -p models/saved_model

# Set environment variables
ENV PYTHONPATH /app

# Command to run the pipeline
CMD ["python", "src/pipeline.py"]

FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# If other deps are needed, install here, for example:
# RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .

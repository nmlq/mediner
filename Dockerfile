# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (if any are required by your requirements.txt)
# This is a placeholder. Replace with actual system dependencies if needed.
# Example:  RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .

# Expose any necessary ports (if your app listens on a port)
# Example: EXPOSE 8000

# Define the command to run your application
# This assumes your entry point is configured correctly in setup.py.
CMD ["mediner"]

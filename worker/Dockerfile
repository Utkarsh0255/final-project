# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install dependencies without saving the cache to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy your research engine and worker code
COPY . .

# Start the worker to listen to the Redis queue
CMD ["python", "worker.py"]
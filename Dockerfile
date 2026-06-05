FROM python:3.9-slim

# Install system dependencies for Matplotlib and Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies and install them (using CPU-only torch to save space)
COPY requirements_web.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements_web.txt

# Copy the rest of the project files
COPY . .

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]

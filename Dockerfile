FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for the API
EXPOSE 8000

# Run the API (assuming FastAPI based on api/main.py)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

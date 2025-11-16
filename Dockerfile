# Stage 1: Builder - Install dependencies
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install Poetry and dependencies
RUN pip install poetry
WORKDIR /app
COPY pyproject.toml poetry.lock /app/

# Install packages
RUN poetry install --no-root --only main

# Stage 2: Final Image - Minimal runtime environment
FROM python:3.10-slim

# Copy virtual environment from builder stage
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /root/.cache/pypoetry/virtualenvs/ /root/.cache/pypoetry/virtualenvs/

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Expose the port your server runs on (e.g., 8000 for FastAPI)
EXPOSE 8000

# Command to run the inference server (Example using uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

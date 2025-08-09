# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Create a dedicated user and group to run the application
RUN useradd --create-home --shell /bin/bash appuser

# --- THE FIX ---
# Give the new user ownership of the working directory
RUN chown -R appuser:appuser /app

# Switch to the non-root user before copying files
USER appuser

# Copy the requirements file first to leverage Docker's layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies into the user's local environment
RUN pip install --no-cache-dir --user -r requirements.txt

# Add the user's local bin to the PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy the rest of the application's code
COPY --chown=appuser:appuser . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
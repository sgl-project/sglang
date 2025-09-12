# SGLang Hathora Deployment Dockerfile
# Use a slim Python image to keep the container size small
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install gcloud CLI
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates dnsutils \
  && mkdir -p /usr/share/keyrings \
  && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
  && apt-get update && apt-get install -y --no-install-recommends google-cloud-sdk \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install SGLang - adjust this line based on your SGLang installation method
# For development, you might want to install from source or a specific version
RUN pip install --no-cache-dir sglang[all]

# Copy the application code
COPY app/serve_hathora.py .
COPY app/hathora_config.py .

# Copy entrypoint script
COPY app/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

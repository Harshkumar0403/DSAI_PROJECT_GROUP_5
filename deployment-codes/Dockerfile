FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch (AUTO GPU/CPU COMPATIBLE)
RUN pip install --no-cache-dir --user torch torchvision

# Install rest
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy app
COPY --chown=user . .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "main.py"]
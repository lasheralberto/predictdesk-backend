# Stage 1: build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Copiar requirements sin torch/torchvision/torchaudio
COPY requirements.txt .

# Instalar torch, torchvision y torchaudio desde índice oficial de PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

    
# Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: imagen final
FROM python:3.10-slim

WORKDIR /app

# Copiar dependencias del builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar el código de la app
COPY . .

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

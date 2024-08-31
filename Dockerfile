FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

RUN pip3 install fastapi[standard]

CMD [ "fastapi", "run", "--host", "0.0.0.0", "--port", "8888" ]
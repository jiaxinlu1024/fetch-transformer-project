# Use PyTorch CPU version
FROM pytorch/pytorch:2.5.1-cpu

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]

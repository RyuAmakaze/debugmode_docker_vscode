# Use a PyTorch image with CUDA support so GPU acceleration is available.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies needed for building qiskit-aer with CUDA
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git build-essential cmake libopenblas-dev tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required Python packages (except qiskit-aer, we'll build that manually)
RUN pip install --no-cache-dir \
    qiskit==1.2.0 \
    pytest==8.3.5 \
    matplotlib \
    torchvision==0.16.0 \
    tqdm \
    pylatexenc \
    qiskit-aer-gpu \
    xformers \
    debugpy \
    dotenv

# Copy your project code
COPY . /app

# Default command
CMD ["python", "src/run.py"]
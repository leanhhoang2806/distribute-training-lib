# Use the Ubuntu 20.04 base image
FROM ubuntu:20.04

# Set the environment variables for CUDA version and cuDNN version
ENV CUDA_VERSION=11.0
ENV CUDNN_VERSION=8.0.5.39-1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA drivers and CUDA toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda=${CUDA_VERSION}-1 \
    libcudnn8=${CUDNN_VERSION}+cuda${CUDA_VERSION} \
    libcudnn8-dev=${CUDNN_VERSION}+cuda${CUDA_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"

# Set the working directory inside the container
WORKDIR /app

# Copy the Python app files to the container's working directory
COPY . /app

# Install any Python dependencies (if needed)
# RUN pip install -r requirements.txt

# Execute the unittest command inside the container
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]
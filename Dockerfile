# Use the Ubuntu 20.04 base image
FROM ubuntu:20.04

# Set the environment variables for CUDA version and cuDNN version
ENV CUDA_VERSION=11.4.2
ENV CUDNN_VERSION=8.2.0.53

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Add the NVIDIA CUDA repository
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add -
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Install CUDA and cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda=${CUDA_VERSION}-1 \
    libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA_VERSION} \
    libcudnn8-dev=${CUDNN_VERSION}-1+cuda${CUDA_VERSION} \
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

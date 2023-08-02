# Use the Ubuntu 20.04 base image
FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
# Set the working directory inside the container
# Use the NVIDIA CUDA 11.2.1 base image with Ubuntu 20.04


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# Set the working directory inside the container
WORKDIR /app

# Copy the Python app files to the container's working directory
COPY . /app

# Install any Python dependencies (if needed)
# RUN pip install -r requirements.txt

# Execute the unittest command inside the container
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

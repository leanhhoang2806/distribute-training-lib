# Use the Ubuntu 20.04 base image with CUDA support
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install Python 3.8 and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    build-essential
RUN rm -rf /var/lib/apt/lists/*

# Set up Python environment and aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy the Python app files to the container's working directory
COPY . /app

# Install any Python dependencies (if needed)
RUN pip install -r requirements.txt

# Execute the unittest command inside the container
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

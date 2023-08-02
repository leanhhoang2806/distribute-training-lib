# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu as Builder

# Install Conda
# Install required packages for downloading Miniconda
# Install PyCUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client

FROM tensorflow/tensorflow:latest-gpu as PipInstaller

# Copy system-level packages from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda


# Set the working directory inside the container
WORKDIR /app


# Copy the Python app files to the container's working directory
COPY . /app

# Install any Python dependencies (if needed)
RUN pip install -r requirements.txt

# Execute the unittest command inside the container
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

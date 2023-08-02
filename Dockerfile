# Use the Ubuntu 20.04 base image
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
RUN nvidia-smi
# Set the working directory inside the container
WORKDIR /app

# Copy the Python app files to the container's working directory
COPY . /app

# Install any Python dependencies (if needed)
# RUN pip install -r requirements.txt

# Execute the unittest command inside the container
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

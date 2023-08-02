# Use the Ubuntu 20.04 base image
FROM ubuntu:20.04

RUN sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit-base

RUN nvidia-ctk --version
# Set the working directory inside the container
WORKDIR /app

# Copy the Python app files to the container's working directory
COPY . /app

# Install any Python dependencies (if needed)
# RUN pip install -r requirements.txt

# Execute the unittest command inside the container
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

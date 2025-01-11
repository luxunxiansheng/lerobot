# Use a clean Ubuntu image
FROM ubuntu:22.04

# Install basic dependencies (optional)
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    curl \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set PATH for Conda
ENV PATH="/opt/conda/bin:$PATH"

# Create and activate default Conda environment
RUN conda create -n lerobot python=3.10 -y && \
    echo "conda activate lerobot" >> ~/.bashrc
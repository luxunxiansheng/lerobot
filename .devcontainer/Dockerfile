FROM nvcr.io/nvidia/l4t-base:r36.4.3

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y sudo && \
    apt-get install -y screen && \
    apt-get install -y --no-install-recommends git && \
    apt-get install -y build-essential && \
    apt-get install wget -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


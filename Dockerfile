# Use Python 3.11 as base image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3-dev \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install -y imagemagick

RUN apt-get update && apt-get install -y wget && \
    apt-get install -y autoconf pkg-config

RUN apt-get update && apt-get install -y wget && \
    apt-get install -y build-essential curl libpng-dev && \
    wget https://github.com/ImageMagick/ImageMagick/archive/refs/tags/7.1.0-31.tar.gz && \
    tar xzf 7.1.0-31.tar.gz && \
    rm 7.1.0-31.tar.gz && \
    apt-get clean && \
    apt-get autoremove

RUN apt-get update && \
    apt-get -y install ghostscript

RUN sh ./ImageMagick-7.1.0-31/configure --prefix=/usr/local --with-bzlib=yes --with-fontconfig=yes --with-freetype=yes --with-gslib=yes --with-gvc=yes --with-jpeg=yes --with-jp2=yes --with-png=yes --with-tiff=yes --with-xml=yes --with-gs-font-dir=yes && \
    make -j && make install && ldconfig /usr/local/lib/

# Install Python dependencies and the package itself
RUN pip install -r requirements.txt \
    && pip install --no-cache-dir -e .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the application
CMD ["streamlit", "run", "1_🏠_Home.py"]

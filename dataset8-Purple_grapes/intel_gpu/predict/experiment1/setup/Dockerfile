FROM openvino/ubuntu24_runtime
ENV DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    clinfo \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip and setuptools ---
RUN python3 -m pip install --upgrade pip setuptools

# --- Copy requirements and install Python dependencies ---
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# --- Override PyTorch and install Intel Extension ---
RUN python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/xpu
RUN python3 -m pip install intel-extension-for-pytorch==2.6.10+xpu oneccl_bind_pt==2.6.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# --- Fix numpy._core error ---
RUN pip uninstall -y numpy && pip install numpy --no-cache-dir

# --- Copy the rest of your application ---
COPY . .

# --- Default command ---
CMD sh -c "python3 yolo.py --eval --batch_size 1 --device intel:gpu > results.txt 2>&1"

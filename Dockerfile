# Multi-runtime Dockerfile for Deaf-Mute Assistant
# Supports: Python 3.12, Node.js, Bun, Rust
# Target: Raspberry Pi 5 (ARM64) and development

FROM debian:bookworm-slim AS base

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (without Python 3.12 - not in Bookworm repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    pkg-config \
    cmake \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    # Python build dependencies (for compiling Python 3.12)
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    libncurses5-dev \
    libreadline-dev \
    liblzma-dev \
    zlib1g-dev \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Video/Camera support
    v4l-utils \
    libv4l-dev \
    # Tauri/GUI dependencies
    libwebkit2gtk-4.1-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 from source (Debian Bookworm only has 3.11)
ENV PYTHON_VERSION=3.12.4
RUN cd /tmp \
    && wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j$(nproc) \
    && make altinstall \
    && cd / && rm -rf /tmp/Python-${PYTHON_VERSION}* \
    && ln -sf /usr/local/bin/python3.12 /usr/bin/python3.12 \
    && ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

# ---------------------------------------------------------------------------
# Node.js (via fnm for flexibility)
# ---------------------------------------------------------------------------
ENV NODE_VERSION=22
ENV FNM_DIR=/usr/local/fnm
ENV PATH="${FNM_DIR}:${PATH}"

RUN curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "${FNM_DIR}" --skip-shell \
    && eval "$(${FNM_DIR}/fnm env)" \
    && fnm install ${NODE_VERSION} \
    && fnm default ${NODE_VERSION} \
    && fnm alias default system

ENV PATH="${FNM_DIR}/aliases/default/bin:${PATH}"

# ---------------------------------------------------------------------------
# Bun
# ---------------------------------------------------------------------------
ENV BUN_INSTALL=/usr/local/bun
ENV PATH="${BUN_INSTALL}/bin:${PATH}"

RUN curl -fsSL https://bun.sh/install | bash

# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rustup target add aarch64-unknown-linux-gnu

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy the entire project
COPY . .

# ---------------------------------------------------------------------------
# Python setup (hand-detector)
# ---------------------------------------------------------------------------
RUN python -m pip install --break-system-packages --upgrade pip \
    && python -m pip install --break-system-packages \
    ./apps/hand-detector/.wheels/mediapipe-0.10.35-cp312-cp312-linux_aarch64.whl \
    && python -m pip install --break-system-packages -r apps/hand-detector/requirements.txt

# ---------------------------------------------------------------------------
# Node/Bun setup
# ---------------------------------------------------------------------------
RUN bun install

# Default command
CMD ["bash"]

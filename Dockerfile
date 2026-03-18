# ── Base image: CUDA 11.8 + cuDNN 8 (matches TensorFlow 2.12–2.13) ────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3       /usr/bin/pip

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter notebook

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Jupyter config: disable token auth for local dev ─────────────────────────
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py

# ── Expose Jupyter port ───────────────────────────────────────────────────────
EXPOSE 8888

# ── Default command: start Jupyter ───────────────────────────────────────────
CMD ["jupyter", "notebook", "--notebook-dir=/workspace/notebooks", "--allow-root"]

# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    build-essential \
    cmake \
    libopenblas-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Install custom nodes via comfy-node-install (ComfyUI registry)
RUN comfy-node-install \
    comfyui-videohelpersuite \
    comfyui-frame-interpolation \
    comfyui-mixlab-nodes

# Install comfyui-reactor-node (Codeberg, not in ComfyUI registry)
RUN comfy --workspace /comfyui node install --git-url https://codeberg.org/Gourieff/comfyui-reactor-node \
    && uv pip install insightface "onnxruntime-gpu==1.18.0"

# Download ReactorNode required face models
RUN comfy --workspace /comfyui model download \
        --url "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx" \
        --relative-path models/insightface \
    && comfy --workspace /comfyui model download \
        --url "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l.zip" \
        --relative-path models/insightface \
    && python -c "import zipfile; zipfile.ZipFile('/comfyui/models/insightface/buffalo_l.zip').extractall('/comfyui/models/insightface/models/')" \
    && rm /comfyui/models/insightface/buffalo_l.zip \
    && comfy --workspace /comfyui model download \
        --url "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth" \
        --relative-path models/facerestore_models \
    && comfy --workspace /comfyui model download \
        --url "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth" \
        --relative-path models/facerestore_models

# Download FILM VFI model
RUN comfy --workspace /comfyui model download \
    --url "https://huggingface.co/jkawamoto/frame-interpolation-pytorch/resolve/main/film_net_fp32.pt" \
    --relative-path models/frame_interpolation/FILM

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=none

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/text_encoders models/diffusion_models models/model_patches

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      comfy --workspace /comfyui model download \
          --url https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors \
          --relative-path models/checkpoints \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors \
          --relative-path models/vae \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors \
          --relative-path models/vae \
          --filename sdxl-vae-fp16-fix.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      comfy --workspace /comfyui model download \
          --url https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors \
          --relative-path models/checkpoints \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}"; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      comfy --workspace /comfyui model download \
          --url https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors \
          --relative-path models/unet \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}" \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors \
          --relative-path models/clip \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
          --relative-path models/clip \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors \
          --relative-path models/vae \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}"; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      comfy --workspace /comfyui model download \
          --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors \
          --relative-path models/unet \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}" \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors \
          --relative-path models/clip \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
          --relative-path models/clip \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors \
          --relative-path models/vae \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}"; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      comfy --workspace /comfyui model download \
          --url https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors \
          --relative-path models/checkpoints; \
    fi

RUN if [ "$MODEL_TYPE" = "z-image-turbo" ]; then \
      comfy --workspace /comfyui model download \
          --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors \
          --relative-path models/text_encoders \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}" \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
          --relative-path models/diffusion_models \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}" \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors \
          --relative-path models/vae \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}" \
      && comfy --workspace /comfyui model download \
          --url https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors \
          --relative-path models/model_patches \
          --hf-token "${HUGGINGFACE_ACCESS_TOKEN}"; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

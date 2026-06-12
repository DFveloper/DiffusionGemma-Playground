# Builds llama-diffusion-cli from llama.cpp PR #24423 (DiffusionGemma, unmerged draft)
# RTX 5090 = Blackwell = CUDA arch 120, needs CUDA >= 12.8
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp . \
 && git fetch --depth 1 origin pull/24423/head:pr-24423 \
 && git checkout pr-24423

RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
 && cmake --build build -j --config Release --target llama-diffusion-cli

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 openssl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/llama-diffusion-cli /usr/local/bin/

ENTRYPOINT ["llama-diffusion-cli"]

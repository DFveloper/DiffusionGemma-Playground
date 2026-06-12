FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        pkg-config \
        ninja-build \
        libcurl4-openssl-dev \
        libssl-dev \
        libvulkan-dev \
        vulkan-tools \
        glslang-tools \
        libshaderc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN git clone --depth 1 https://github.com/DFveloper/aikar-engine.git .

RUN cmake -B build -G Ninja \
        -DGGML_VULKAN=ON \
        -DGGML_NATIVE=ON \
        -DBUILD_SHARED_LIBS=OFF \
 && cmake --build build -j --target llama-diffusion-cli

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        openssl \
        libvulkan1 \
        libshaderc1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/llama-diffusion-cli /usr/local/bin/

ENTRYPOINT ["llama-diffusion-cli"]

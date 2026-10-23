# 설치하기

이 안내에서는 `cargo oxide run vecadd`를 실행하는 데 필요한 환경을 준비합니다. 최신 요구사항과 상세한 진단 방법은 [영문 설치 안내](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/installation.html)도 함께 확인하세요.

## 필요한 환경

| 항목 | 요구사항 |
|---|---|
| 운영체제 | Linux (Ubuntu 24.04에서 검증, Windows 미지원) |
| GPU | NVIDIA Ampere 이상 (`sm_80+`) |
| 드라이버 | CUDA 13.x 호환 드라이버 580 이상 |
| CUDA Toolkit | 13.0 이상 (`nvcc`, `cuda.h`, `curand.h` 필요) |
| LLVM | 21 이상, NVPTX 백엔드 포함 |
| Clang | 21 이상. `bindgen`이 호스트 CUDA 바인딩을 만들 때 사용 |
| Rust | 저장소의 `cuda-oxide/rust-toolchain.toml`이 지정한 nightly |

드라이버는 실행 중 CUDA 바인딩이 처음 호출될 때 로드됩니다. 따라서 드라이버가 없거나 설치한 Toolkit보다 너무 오래되면 프로그램 시작 뒤 첫 드라이버 호출에서 실패할 수 있습니다. `cargo oxide doctor`를 실행해 환경 진단을 먼저 하세요.

## 개발 컨테이너로 시작하기

저장소에는 CUDA Toolkit, LLVM, Clang과 고정된 Rust nightly를 포함한 개발 컨테이너 설정이 있습니다. 호스트에는 CUDA Toolkit을 따로 설치할 필요가 없지만, NVIDIA GPU와 호환 드라이버, Docker 및 NVIDIA Container Toolkit이 필요합니다.

저장소 루트에서 컨테이너를 시작합니다.

```bash
npx -y @devcontainers/cli up --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json
```

컨테이너에서 환경을 확인하고 예제를 실행합니다.

```bash
npx -y @devcontainers/cli exec --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide doctor
npx -y @devcontainers/cli exec --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide run vecadd
```

호스트 드라이버가 요구 버전보다 오래되면 컨테이너 안에서도 GPU 명령이 실패합니다. 이때는 컨테이너의 Toolkit을 바꾸기보다 호스트 NVIDIA 드라이버를 업데이트하세요.

## Nix 개발 셸 사용하기

Linux와 flakes가 활성화된 Nix를 사용한다면 저장소가 제공하는 개발 셸을 이용할 수 있습니다.

```bash
cd cuda-oxide
nix develop
cargo oxide run vecadd
```

새 프로젝트는 저장소를 직접 복제하지 않고도 만들 수 있습니다.

```bash
nix run 'github:NVIDIA/cuda-rust?dir=cuda-oxide#new' -- my-project
cd my-project && nix develop
```

Nix 셸을 사용한다면 아래의 CUDA, LLVM, Clang, Rust 수동 설치 과정은 건너뛸 수 있습니다. 호스트에는 NVIDIA 드라이버가 필요합니다.

## 수동 설치

### CUDA Toolkit

[NVIDIA CUDA 다운로드](https://developer.nvidia.com/cuda-downloads) 페이지에서 Toolkit 13.0 이상을 설치합니다. 기본 경로가 아니라면 `CUDA_TOOLKIT_PATH`로 `include/cuda.h`가 들어 있는 Toolkit 루트를 지정할 수 있습니다.

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

### LLVM

LLVM의 NVPTX 백엔드는 LLVM IR을 PTX로 낮추는 데 사용됩니다. 먼저 Rust 툴체인에 포함된 `llc`가 있는지 확인하세요. 별도 LLVM을 설치한다면 21 이상이며 NVPTX 대상이 활성화되어 있어야 합니다.

```bash
llc-21 --version | grep nvptx
```

출력에 `nvptx64` 대상이 보여야 합니다. 배포판 패키지에 대상이 없거나 직접 LLVM을 빌드한다면 NVPTX 백엔드를 포함하도록 구성해야 합니다.

### Clang과 Rust

호스트 CUDA 바인딩 생성에는 `bindgen`과 Clang 개발 헤더가 필요합니다. Ubuntu에서는 다음 패키지를 설치합니다.

```bash
sudo apt install libclang-21-dev libclang-cpp21-dev libclang-common-21-dev
```

저장소의 `cuda-oxide/` 디렉터리에서 Cargo 명령을 실행하면 `rust-toolchain.toml`에 고정된 nightly와 구성요소를 rustup이 준비합니다. 직접 설치하려면 해당 파일에 적힌 툴체인 버전을 사용하세요.

### cargo-oxide 설치 및 점검

저장소 밖의 프로젝트에서 사용하려면 고정된 nightly로 `cargo-oxide`를 설치합니다.

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
cargo oxide doctor
```

`doctor`가 호환 GPU, CUDA Toolkit, LLVM, 코드 생성 백엔드를 점검합니다. 진단 결과를 먼저 해결한 다음 [첫 커널 작성하기](hello-gpu.md)를 진행하세요.

## 참고

CUDA Toolkit과 NVIDIA 드라이버의 호환성은 버전 조합에 따라 달라집니다. 특히 드라이버가 새 PTX 버전을 이해하지 못하면 모듈 적재 시 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`이 발생할 수 있습니다. 구체적인 호환 표와 문제 해결 절차는 [원문 설치 안내](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/installation.html#toolkit-and-driver-compatibility)를 확인하세요.

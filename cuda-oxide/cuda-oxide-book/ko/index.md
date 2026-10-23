# cuda-oxide 책 (한국어)

**cuda-oxide**는 일반 Rust 코드를 NVIDIA GPU에서 실행되는 PTX로 컴파일하는 도구입니다. 별도의 GPU 전용 언어를 쓰지 않고 Rust로 커널을 작성하면서 CUDA의 실행 모델과 메모리를 직접 다룹니다.

> NVIDIA [원본 cuda-rust 저장소](https://github.com/NVIDIA/cuda-rust)의 문서를 한국어로 옮기고 있습니다. 원문과 번역이 다르면 코드와 최신 원문을 기준으로 확인해 주세요. 이 프로젝트는 초기 단계여서 도구와 API가 바뀔 수 있습니다.

## 시작하기

- [설치하기](getting-started/installation.md) — 지원 환경과 도구 준비
- [첫 커널 작성하기](getting-started/hello-gpu.md) — 벡터 덧셈 커널을 만들고 실행
- [영문 원본 책](https://nvidia.github.io/cuda-rust/cuda-oxide/)
- [원본 저장소](https://github.com/NVIDIA/cuda-rust)

:::{toctree}
:hidden:
:maxdepth: 2

getting-started/installation
getting-started/hello-gpu
:::

## cuda-oxide란?

cuda-oxide는 Rust 컴파일러 백엔드를 사용해 Rust 코드를 CUDA PTX로 변환합니다. 커널 코드에도 Rust 타입과 소유권을 활용할 수 있으며, 스레드·블록·메모리·동기화 방식은 CUDA 실행 모델에 맞춰 직접 구성합니다.

Rust의 타입 검사는 메모리 접근 실수를 줄이는 데 도움을 주지만, GPU 프로그램의 모든 경쟁 상태나 동기화 오류를 자동으로 없애지는 않습니다. 커널의 실행 범위와 각 스레드가 쓰는 메모리 위치를 올바르게 설계해야 합니다.

## 번역 현황

설치 안내와 첫 커널 안내부터 옮기고 있습니다. 다른 장의 최신 내용은 [영문 원본 목차](https://nvidia.github.io/cuda-rust/cuda-oxide/)에서 확인할 수 있습니다. 번역 오류를 발견하면 [포크 저장소](https://github.com/wkd09/cuda-rust)에 제안해 주세요.

# cuda-oxide 책 (한국어 번역)

**cuda-oxide**는 일반 Rust 코드를 PTX로 컴파일하는 Rust→CUDA 컴파일러입니다. 별도의 DSL이나 다른 언어 바인딩 없이, 익숙한 Rust 코드로 CUDA의 SIMT GPU 커널을 작성할 수 있습니다.

> 이 문서는 NVIDIA의 [cuda-rust 원본 저장소](https://github.com/NVIDIA/cuda-rust)를 바탕으로 한 한국어 번역입니다. 원문과 번역에 차이가 있으면 원문 및 실제 코드가 기준입니다. CUDA Rust는 초기 알파 단계의 연구 프로젝트이므로 기능과 API가 변경될 수 있습니다.

## 시작하기

- [설치 안내 (원문)](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/installation.html)
- [첫 GPU 프로그램 (원문)](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/hello-gpu.html)
- [원본 cuda-oxide 책](https://nvidia.github.io/cuda-rust/cuda-oxide/)
- [원본 저장소](https://github.com/NVIDIA/cuda-rust)

## cuda-oxide는 무엇인가요?

cuda-oxide는 표준 Rust 코드를 CUDA PTX로 컴파일하는 커스텀 `rustc` 코드 생성 백엔드입니다. GPU 코드도 Rust의 타입 시스템과 소유권 모델을 활용해 작성하며, 커널 실행 시 스레드와 블록, 메모리, 동기화 방식을 직접 제어합니다.

프로젝트의 핵심 목표는 GPU 메모리 접근에서 잘못된 동시 쓰기를 줄이는 것입니다. 예를 들어 변경 가능한 슬라이스를 스레드별로 서로 겹치지 않게 나누고, 각 스레드는 자신에게 할당된 원소만 수정하도록 설계할 수 있습니다. 다만 GPU 동기화와 공유 메모리에는 별도의 주의가 필요합니다.

## 문서 번역 현황

현재 한국어 문서는 순차적으로 번역 중입니다. 번역되지 않은 내용은 위의 원문 링크에서 확인해 주세요. 각 번역 페이지는 원본 변경을 주기적으로 확인하고 최신 내용에 맞춰 갱신합니다.

## 원본 문서 목차

전체 주제와 최신 구조는 [원본 목차](https://nvidia.github.io/cuda-rust/cuda-oxide/)를 참고하세요.

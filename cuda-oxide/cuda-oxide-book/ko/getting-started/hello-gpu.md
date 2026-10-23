# 첫 GPU 커널 작성하기

이 장에서는 `cargo-oxide`로 프로젝트를 만들고, 벡터 덧셈 커널을 실행하는 흐름을 살펴봅니다. 코드 생성 세부사항과 비동기 예제는 [영문 원문](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/hello-gpu.html)을 참고하세요.

## 도구 설치와 프로젝트 생성

아직 설치하지 않았다면 고정된 nightly 툴체인으로 `cargo-oxide`를 설치하고, 환경 진단을 실행합니다.

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
cargo oxide doctor
```

새 프로젝트를 만든 뒤 실행합니다.

```bash
cargo oxide new my_first_kernel
cd my_first_kernel
cargo oxide run
```

기본 템플릿은 1024개 원소의 벡터 덧셈을 수행합니다. 성공하면 모든 결과가 맞았다는 메시지가 출력됩니다.

## 커널의 기본 구조

아래 예제에서 `#[kernel]` 함수가 GPU 진입점입니다. 각 스레드는 자신의 1차원 인덱스를 얻고, 출력 슬라이스의 해당 원소가 존재할 때만 값을 기록합니다.

```rust
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
```

`#[cuda_module]`은 커널 모듈을 감싸고 호스트에서 사용할 로더와 타입이 지정된 실행 인터페이스를 생성합니다. 커널에서 호출되는 보조 함수는 별도 표시 없이 함께 찾아 컴파일할 수 있습니다. 실행 설정과 메모리 복사까지 포함한 전체 예제는 [원문 커널 안내](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/hello-gpu.html#anatomy-of-a-kernel)에 있습니다.

## 컴파일과 실행 흐름

cuda-oxide는 한 Rust 소스 파일에서 호스트 코드와 커널을 함께 다룹니다. 커널은 대략 다음 경로로 컴파일됩니다.

```text
Rust 소스 → MIR → Pliron IR → LLVM IR → PTX
```

호스트 코드는 일반 Rust 코드 생성 경로를 따르고, 생성된 GPU 산출물은 실행 파일에 포함됩니다. 런타임에서는 CUDA 컨텍스트와 스트림을 준비하고, 데이터를 GPU 메모리로 복사한 뒤 커널을 실행합니다. 계산 결과를 호스트로 다시 복사해 검증합니다.

## 다음 단계

- [설치 안내](installation.md)에서 지원 환경과 도구 요구사항을 확인하세요.
- 원문에서 [커널 인자와 ABI](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/hello-gpu.html#argument-scalarization), [비동기 실행](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/hello-gpu.html#going-async)도 살펴보세요.
- GPU가 보이지 않거나 툴체인 진단이 실패하면 `cargo oxide doctor` 결과와 원문 문제 해결 절차를 확인하세요.

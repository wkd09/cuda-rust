# cuda-oxide 책

```{image} _static/images/banner-light.png
:alt: cuda-oxide: 순수 Rust로 CUDA (SIMT) 커널 작성하기
:align: center
:width: 640px
:class: only-light mb-4
```

```{image} _static/images/banner-dark.png
:alt: cuda-oxide: 순수 Rust로 CUDA (SIMT) 커널 작성하기
:align: center
:width: 640px
:class: only-dark mb-4
```

**cuda-oxide**는 관용적인 Rust 코드로 (SIMT) GPU 커널을 작성할 수 있게 해 주는 Rust-to-CUDA 컴파일러입니다. 일반 Rust 코드를 PTX로 직접 컴파일하므로 별도의 DSL이나 외국어 바인딩 없이 Rust만 사용합니다. 메모리 안전성을 중요한 목표로 삼지만, GPU 프로그래밍의 세부 사항을 모두 자동으로 안전하게 처리하지는 않습니다.

:::{note}
이 책은 소유권, 트레이트, 제네릭을 비롯한 Rust 언어의 기본 지식을 갖추고 있다고 가정합니다. 뒤쪽의 비동기 GPU 프로그래밍 장에서는 `async`/`.await`와 tokio 같은 런타임에 대한 이해도 필요합니다.

복습이 필요하다면 [The Rust Programming Language](https://doc.rust-lang.org/book/), [Rust by Example](https://doc.rust-lang.org/rust-by-example/), [Async Book](https://rust-lang.github.io/async-book/)을 참고하세요.
:::

---

## 프로젝트 상태

v0.1.0은 초기 알파 릴리스입니다. 프로젝트를 개선하는 동안 **버그, 미완성 기능, API 호환성 변경이 있을 수 있습니다.** 직접 사용해 보고 경험과 의견을 공유해 프로젝트의 방향을 함께 만들어 주세요.

---

## 🚀 빠른 시작

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // SAFETY: 이 패키지는 `kernels`용 디바이스 번들을 실행 파일에 포함합니다.
    let module = unsafe { kernels::load(&ctx).unwrap() };

    let a = DeviceBuffer::from_host(&stream, &[1.0f32; 1024]).unwrap();
    let b = DeviceBuffer::from_host(&stream, &[2.0f32; 1024]).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    let prepared = module
        .prepare_vecadd(LaunchConfig1D::new(1024u32.div_ceil(256), 256, 0))
        .unwrap();
    module
        .vecadd(&stream, &prepared, &a, &b, &mut c)
        .unwrap();

    let result = c.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 3.0);
}
```

[필수 환경](getting-started/installation.md)을 준비한 다음 `cargo oxide run vecadd`로 빌드하고 실행하세요. 같은 launch-contract 패턴은 `cargo oxide new`가 만드는 프로젝트의 기본 구조이기도 합니다. 자세한 설명은 [첫 GPU 커널 작성하기](getting-started/hello-gpu.md)를 참고하세요.

:::{note}
`#[cuda_module]`은 생성된 디바이스 산출물을 호스트 실행 파일에 포함하고 타입이 지정된 `kernels::load` 함수와 커널마다 하나의 실행 메서드를 생성합니다. 커널 인자는 타입 검사를 거칩니다. `#[launch_contract]`를 선언하면 [커널 실행](https://nvidia.github.io/cuda-rust/cuda-oxide/gpu-programming/launching-kernels.html) 장에서 다루는 안전한 `PreparedLaunch` 경로(`prepare_*`와 타입이 지정된 실행 메서드)를 사용할 수 있습니다. 계약에서 다루지 않는 일회성 실행 형상이 필요하면 원시 `LaunchConfig`를 사용하는 unsafe 호출도 가능합니다. 특정 sidecar 산출물을 불러오거나 사용자 지정 실행 코드를 만들어야 한다면 더 낮은 수준의 `load_kernel_module` 및 unsafe `cuda_launch!` API도 사용할 수 있습니다.
:::

---

## cuda-oxide를 선택하는 이유

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🦀 GPU에서 Rust 사용하기
Rust의 타입 시스템과 소유권 모델로 GPU 커널을 작성합니다. 안전성을 중요한 목표로 삼되 GPU에는 별도의 주의 사항이 있으므로 [GPU 안전성 모델](https://nvidia.github.io/cuda-rust/cuda-oxide/gpu-safety/the-safety-model.html)을 확인하세요.
:::

:::{grid-item-card} 💎 SIMT 컴파일러
DSL이 아닙니다. 커스텀 `rustc` 코드 생성 백엔드가 순수 Rust를 PTX로 컴파일합니다.
:::

:::{grid-item-card} ⚡ 비동기 실행
지연 실행되는 `DeviceOperation` 그래프를 조합합니다. 스트림 풀에 작업을 예약하고 `.await`로 결과를 기다립니다.
:::

::::

## 책 목차

현재 번역된 장은 아래에서 읽을 수 있습니다. 번역되지 않은 장의 최신 원문은 [영문 cuda-oxide 책](https://nvidia.github.io/cuda-rust/cuda-oxide/)에서 확인하세요.

:::{toctree}
:hidden:
:maxdepth: 2
:caption: 시작하기

getting-started/installation
getting-started/hello-gpu
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: 고급 GPU 기능

advanced/cluster-programming
advanced/matrix-multiply-accelerators
advanced/tensor-memory-accelerator
advanced/warp-level-programming
advanced/shared-memory-and-synchronization
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: GPU 프로그래밍

gpu-programming/execution-model
gpu-programming/closures-and-generics
gpu-programming/virtual-memory-and-peer-access
gpu-programming/kernels-and-device-functions
gpu-programming/error-handling-and-debugging
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: 비동기 프로그래밍

async-programming/overlapping-transfers-and-compute
async-programming/scheduling-and-streams
async-programming/combinators-and-composition
async-programming/the-device-operation-model
async-programming/concurrent-execution
:::

:::{toctree}
:hidden:
:maxdepth: 1
:caption: 부록

appendix/building-from-source
appendix/cuda-cpp-comparison
appendix/ecosystem
appendix/supported-features
appendix/glossary
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: GPU 안전성

gpu-safety/bounds-checks
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: 컴파일러 내부

compiler/fuzzing-and-differential-testing
compiler/rustc-public
compiler/pliron
compiler/rustc-codegen-cuda
compiler/architecture-overview
compiler/compiler-optimizations
compiler/catalog-generated-intrinsics
compiler/mlir-dialects
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: 프로젝트

projects/async-mlp-pipeline
:::

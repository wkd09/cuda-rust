# 첫 GPU 커널 작성하기

이 절에서는 cuda-oxide 설치부터 프로젝트 생성, GPU 커널 작성과 실행까지 순수 Rust로 진행합니다.

---

## cargo-oxide 설치하기

아직 설치하지 않았다면 고정된 Nightly 도구 체인으로 빌드 도구를 설치합니다.

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
```

환경이 올바르게 준비되었는지 확인합니다:

```bash
cargo oxide doctor
```

이 명령은 호환 GPU, CUDA Toolkit, LLVM, 코드 생성 백엔드를 확인합니다. 보고된 문제를 해결한 다음 진행하세요. 자세한 내용은 [설치 안내](installation.md)를 참조하세요.

---

## 프로젝트 만들기

`cargo oxide new`로 새 프로젝트 뼈대를 만듭니다.

```bash
cargo oxide new my_first_kernel
cd my_first_kernel
```

이 명령은 바로 실행할 수 있는 프로젝트를 생성합니다.

```text
my_first_kernel/
├── Cargo.toml          # dependencies on cuda-device, cuda-host, cuda-core
├── README.md           # doctor + run instructions
├── .gitignore          # target/ and generated device artifacts
├── rust-toolchain.toml # pins the required nightly toolchain
└── src/
    └── main.rs          # kernel + host code in one file
```

프로젝트를 빌드하고 실행합니다.

```bash
cargo oxide run
```

`PASSED: all 1024 elements correct`라는 메시지가 표시됩니다. 생성된 템플릿은 벡터 덧셈 커널입니다. 좋은 출발점이지만, 조금 더 흥미로운 예제를 살펴보겠습니다.

---

## 커널의 구조

다음은 한 가지 변형을 넣은 벡터 덧셈입니다. 원소별 덧셈을 일반 보조 함수로 분리했습니다. 커널과 보조 함수는 호스트 코드와 함께 같은 파일에 있습니다.

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

/// Plain helper function -- no annotation needed.
/// The compiler discovers it automatically because `vecadd` calls it.
fn add(a: f32, b: f32) -> f32 {
    a + b
}

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
            *c_elem = add(a[i], b[i]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
    module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

    let c_host = c_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
    Ok(())
}
```

여기에는 여러 요소가 함께 동작합니다. 핵심 부분을 하나씩 살펴보겠습니다.

### 단일 소스 컴파일

커널과 호스트 코드는 **같은 파일**에 있으며 `cargo` 명령 한 번으로 컴파일됩니다. 코드 생성 백엔드는 컴파일을 가로채 `#[kernel]` 함수를 MIR-to-PTX 파이프라인으로 보내고 나머지는 표준 LLVM에 위임합니다. 최종 바이너리에는 네이티브 호스트 코드와 임베드된 장치 산출물이 함께 들어갑니다.

### `#[kernel]`

함수를 GPU에서 실행할 수 있는 **커널 진입점**, 즉 GPU의 `main`으로 표시합니다. 함수는 다음 파이프라인을 통해 PTX로 컴파일됩니다.

```text
Rust source → MIR → Pliron IR → LLVM IR → PTX
```

같은 함수는 타입 검사를 위해 호스트 컴파일러에도 보이지만 본문은 CPU에서 호출되지 않습니다.

### 장치 함수 자동 탐색

위의 `add` 보조 함수에는 **어떤 어노테이션도 없습니다**. 컴파일러는 `#[kernel]`을 처리할 때 호출 그래프를 따라 커널이 호출하는 모든 함수를 자동으로 찾습니다. 해당 함수들은 장치 함수로 PTX에 컴파일되고 백엔드가 인라인합니다. 보조 함수에 별도 표시를 할 필요가 없습니다.

:::{note}
`#[device]` 어트리뷰트는 존재하지만 용도가 다릅니다. C++에서 사용하는 Rust 장치 라이브러리를 만들 때 독립된 장치 컴파일 루트로 함수를 표시하거나, CUDA C++ LTOIR과 FFI를 위해 `#[device] extern "C" { ... }` 블록에 외부 장치 함수를 선언할 때 사용합니다. 커널에서 호출되는 비공개 보조 함수에는 `#[device]`가 필요하지 않습니다.
:::

### `#[cuda_module]`

`#[cuda_module]`은 인라인 커널 모듈을 감싸고 타입이 지정된 호스트 API를 생성합니다.

```rust
// SAFETY: this package owns the embedded device bundle produced for the
// kernels module above.
let module = unsafe { kernels::load(&ctx)? };
let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;
```

로더는 호스트 바이너리에 임베드된 장치 산출물을 읽고 커널 함수 핸들을 캐시하며 각 `#[kernel]`을 Rust 메서드로 제공합니다. 메서드 서명은 커널 서명을 따르고 장치 슬라이스는 `DeviceBuffer` 참조로 대응됩니다. `vecadd`가 `#[launch_contract]`를 선언했으므로 모듈은 `prepare_vecadd`도 생성합니다. 이 메서드는 차원을 보존하는 `LaunchConfig1D`를 받아 계약과 실제 장치 한도를 검사하고, 안전한 `vecadd` 메서드가 소비하는 `PreparedLaunch<vecadd>` 증거를 반환합니다. 임베드된 번들이 모듈이 선언한 ABI와 일치한다고 호출자가 보증하는 모듈 로딩만이 남은 unsafe 단계입니다. 계약이 없는 커널은 원시 `LaunchConfig`가 1차원 인덱싱 커널에 1차원 실행이 전달됐다는 사실을 증명하지 못하므로 unsafe 원시 실행 메서드만 제공합니다. 전체 준비 실행 경로는 [커널 실행](https://nvidia.github.io/cuda-rust/cuda-oxide/gpu-programming/launching-kernels.html)을 참조하세요.

수동 sidecar 산출물 로딩과 사용자 정의 실행 코드를 위한 저수준 API `load_kernel_module` 및 `cuda_launch!`도 계속 사용할 수 있습니다. `cuda_launch!`는 커널 서명과 인자를 대조할 수 없으므로 unsafe이며 `unsafe { }` 안에서 호출해야 합니다.

### 인자 스칼라화

슬라이스는 호스트/장치 ABI 경계를 `(ptr, len)` 구성요소로 통과합니다. 호스트가 커널 인자 두 개로 전달하고 장치 컴파일러가 진입 블록에서 슬라이스를 다시 구성합니다. 값으로 전달되는 구조체와 클로저는 하나의 byval `.param`으로 전달되므로 호스트 패킷도 집합 전체를 단일 슬롯으로 전달합니다. 이는 실행기가 실제로 동작하는 방식과 일치하며 필드 단위 선언과의 불일치를 방지합니다. 이 과정은 모두 투명하게 처리되어 커널 서명은 일반 Rust와 동일하게 보입니다.

```text
Host:   module.vecadd(&stream, &prepared, &data, ...)
          → extracts (ptr, len) for the slice, passes two args

PTX:    .entry kernel(.param .u64 ptr, .param .u64 len, ...)
          → receives flat slice parameters

Device: kernel body sees unified &[T] slice
          → compiler reconstructs at entry
```

### 구조체의 동적 레이아웃

구조체를 GPU에 전달하면 cuda-oxide는 각 필드의 **정확한 바이트 오프셋**을 rustc에 조회하고 장치 측에서 명시적인 패딩을 넣어 레이아웃을 다시 구성합니다. 따라서 `#[repr(C)]`가 **필수는 아닙니다**. 일반 Rust 구조체도 HMM(GPU의 호스트 메모리 직접 접근)을 포함해 그대로 사용할 수 있습니다.

---

## 비동기 실행

여러 커널로 이루어진 파이프라인이나 동시 작업을 위해 cuda-oxide는 Tokio 기반 비동기 실행 모델을 제공합니다. 비동기 프로젝트를 만들고 동기 방식과의 차이를 살펴보겠습니다.

### 비동기 프로젝트 만들기

```bash
cargo oxide new my_async_kernel --async
cd my_async_kernel
cargo oxide run
```

`--async` 플래그는 `tokio`와 `cuda-async` 의존성이 미리 구성된 프로젝트를 생성합니다.

### 전체 예제

다음은 생성된 비동기 벡터 덧셈 템플릿입니다. 읽기 쉽도록 일부 서식만 조정했습니다.

```rust
use cuda_async::simt::device_context::init_device_contexts;
use cuda_async::simt::device_operation::DeviceOperation;
use cuda_core::simt::LaunchConfig;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_box::DeviceBox;
    use cuda_core::simt::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    // 1. Initialize the device context map (default device 0, 1 device).
    //    The round-robin stream pool is created lazily on first use.
    init_device_contexts(0, 1)?;

    // 2. Load the embedded kernel module from the async device context.
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    // 3. Allocate device memory and copy host data.
    let (a_dev, b_dev, mut c_dev) =
        cuda_async::simt::device_context::with_cuda_context(0, |ctx| {
            let stream = ctx.default_stream();
            let num_bytes = N * mem::size_of::<f32>();
            unsafe {
                let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
                memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
                stream.synchronize().unwrap();
                (
                    DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
                )
            }
        })?;

    // 4. Launch -- returns a lazy DeviceOperation, no GPU work yet.
    // SAFETY: this is a 1D launch and `vecadd` guards its index against the
    // output length before writing.
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .sync()?;  // Block until the GPU finishes.

    // 5. Copy results back to host.
    let mut c_host = vec![0.0f32; N];
    cuda_async::simt::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    // 6. Verify.
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
```

### 동기 방식에서 달라진 점

커널 자체는 **동일합니다**. 비동기 방식은 호스트에서 GPU 작업을 실행하고 관리하는 방법만 바꿉니다.

`{kernel}_async` 메서드
: 즉시 실행하는 대신 지연 상태의 `DeviceOperation`을 반환합니다. 명시적으로 스케줄링하기 전에는 GPU 작업이 일어나지 않습니다. 따라서 리소스를 확정하기 전에 계산 그래프를 구성할 수 있습니다.

`init_device_contexts(default_device, num_devices)`
: 스레드 로컬 장치 컨텍스트 맵을 초기화하고 기본 GPU 번호와 다중 GPU 사용 용량을 설정합니다. 라운드 로빈 스트림 풀은 처음 사용할 때 생성됩니다. 이후 작업은 수동으로 스트림을 관리하지 않아도 라운드 로빈 방식으로 스트림에 배정되어 GPU 점유율을 높입니다.

`DeviceBuffer` 대신 `DeviceBox`
: 스트림 풀과 함께 사용할 수 있는 비동기 안전 장치 메모리 래퍼입니다. `malloc_async`를 사용한 비동기 할당을 지원합니다.

`.sync()`와 `.await`
: `.sync()`는 GPU 작업이 끝날 때까지 호출 스레드를 차단합니다. 호스트에서 다른 할 일이 없을 때 사용하세요. `.await`는 현재 Tokio 작업을 일시 중단해 기다리는 동안 다른 작업을 진행하게 합니다. 호스트에서 동시 작업을 수행하거나 GPU 파이프라인 여러 개가 진행 중일 때 사용하세요.

`and_then`과 `zip!`
: `.and_then(|result| next_op)`로 의존 작업을 연결합니다. 독립 작업은 `zip!(op_a, op_b)`로 동시에 실행합니다. 두 작업을 모두 스트림 풀에 제출하고 둘 다 끝나면 결합된 결과를 사용할 수 있습니다. 이 조합자를 사용하면 복잡한 여러 커널 파이프라인을 선언적으로 표현할 수 있습니다.

:::{tip}
더 완전한 비동기 예제는 `async_mlp`를 참조하세요. `and_then` 연결, 병렬 할당을 위한 `zip!`, 동시 배치 간 `Arc` 공유 가중치를 사용해 GEMM, MatVec, ReLU를 순방향 실행하는 다중 커널 예제입니다. cuda-oxide 작업공간에서 `cargo oxide run async_mlp`로 실행하세요.
:::

# 프로젝트: 비동기 MLP 파이프라인

커널 정의, 데이터 이동, 조합자로 연산 구성, 스트림 간 작업 스케줄링을 배웠습니다. 이제 모든 내용을 하나로 모아 보겠습니다.

이 장에서는 cuda-oxide 저장소에서 가장 완전한 예제인 `async_mlp`를 첫 `use` 구문부터 표준 출력의 마지막 `[ReLU OK]`까지 살펴봅니다. 끝나면 도구 모음의 비동기 패턴을 모두 활용해 네 배치를 동시에 처리하는 멀티 커널 GPU 파이프라인을 만들게 됩니다. 앞 장부터 순서대로 읽었다면 마지막 보스에 해당하는 예제입니다.

:::{seealso}
[동시 실행](../async-programming/concurrent-execution.md) 장에서는 동시 파이프라인의 *개념*을 자세히 설명합니다. 이 장은 실행, 수정, 프로파일링이 가능한 완전한 프로젝트 *코드*에 집중합니다.
:::

---

## 만들 내용

GPU에서 실행하는 간단한 **다층 퍼셉트론(MLP) 순전파**입니다.

```text
input [64×64]  ──►  GEMM(input, W0)  ──►  hidden [64×64]
                                              │
                            MatVec(hidden, W1) ──►  output [64]
                                                       │
                                                 ReLU(output) ──►  result [64]
```

커널 3개, 단계 4개(마지막 디바이스-호스트 복사 포함), 배치 4개, CUDA 스트림 4개, 공유 가중치 한 세트를 사용하며 대기하는 호스트 스레드는 없습니다. 전체 구조는 다음과 같습니다.

```{figure} ../async-programming/images/concurrent-batches.svg
:align: center
:width: 100%

MLP 순전파 네 개가 동시에 실행됩니다. 공유 가중치는 `Arc<DeviceBox>`로 한 번만 업로드한 뒤 각 배치에서 저렴하게 복제합니다. 라운드 로빈 스케줄러는 배치를 네 스트림에 분배하고 파이프라인은 GPU 타임라인에서 서로 겹쳐 실행됩니다.
```

순차 방식은 한 번에 배치 하나를 처리하므로 작업도 대기 시간도 네 배입니다. 동시 방식은 배치 작업을 겹쳐 실행합니다. GPU에 남는 SM이 있다면 단일 배치 처리 시간에 약간의 시작 간격을 더한 정도로 마칠 수 있습니다. `Arc::clone()` 몇 번의 비용에 비하면 괜찮은 결과입니다.

---

## 프로젝트 구조

예제는 `cuda-oxide/crates/rustc-codegen-cuda/examples/async_mlp/`에 있으며 독립 Cargo 워크스페이스 구성원입니다.

```text
async_mlp/
├── Cargo.toml        ← dependencies: cuda-device, cuda-core, cuda-async, tokio
└── src/
    └── main.rs       ← kernels + host code, single file
```

디바이스 커널과 호스트 오케스트레이션이 모두 한 파일에 들어 있습니다. `#[kernel]` 속성은 어떤 함수를 PTX로 만들지 컴파일러에 알려주며 나머지는 일반 Rust로 컴파일됩니다. 별도 `.cu` 파일이나 복잡한 헤더 작업, 서로 분리된 빌드 시스템은 없습니다.

### 의존성

`Cargo.toml`은 필요한 크레이트만 가져옵니다.

```toml
[dependencies]
cuda-device   = { path = "../../../cuda-device" }       # #[kernel], DisjointSlice, thread::*
cuda-core     = "0.3.1"                                 # shared with cutile-rs: CudaModule, simt::LaunchConfig
cuda-async    = "0.3.1"                                 # shared with cutile-rs: simt::DeviceOperation, zip!, and_then
tokio       = { version = "1", features = ["rt", "macros"] }
```

`cuda-device`는 디바이스 API(내장 기능, 안전한 가변 슬라이스)를 제공합니다. `cuda-core`는 모듈 로딩과 실행 설정을 담당합니다. `cuda-async`는 `DeviceOperation` 그래프, 조합자, 스트림 풀 스케줄러를 제공합니다. `tokio`는 퓨처를 폴링하는 호스트 비동기 런타임입니다.

---

## GPU 커널

이 세 함수는 `rustc-codegen-cuda`가 PTX로 컴파일합니다. 호스트에서 실행되지는 않습니다. `#[kernel]` 속성이 각각의 이름을 `cuda_oxide_kernel_<hash>_<name>`으로 바꾸어 코드 생성 백엔드가 식별하고 추출할 수 있게 합니다.

### sgemm_naive — 행렬 곱셈

```rust
use cuda_device::{DisjointSlice, kernel, thread};

use cuda_device::thread::Runtime2DIndex;

#[kernel]
pub fn sgemm_naive(
    m: u32, n: u32, k: u32,
    alpha: f32, a: &[f32], b: &[f32],
    beta: f32, mut c: DisjointSlice<f32, Runtime2DIndex>,
) {
    let row = thread::index_2d_row();
    let col = thread::index_2d_col();

    // The row width comes from `c`, bound on the host to this same `n`.
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        // col < n guaranteed by `Some` -- no manual check needed
        if row < m as usize {
            let n_sz = n as usize;
            let k_sz = k as usize;
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i < k_sz {
                sum = sum + a[row * k_sz + i] * b[i * n_sz + col];
                i = i + 1;
            }
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
```

각 스레드는 출력 행렬의 원소 하나를 계산합니다. 2차원 스레드 인덱스가 (행, 열) 위치에 직접 대응합니다. `DisjointSlice`는 범위를 검사하고 대응하는 인덱스 공간 타입을 요구합니다. `c`의 행 너비는 호출마다 전달하는 인자가 아닙니다. 호스트가 실행 시 슬라이스에 한 번 바인딩하고 `index_2d_runtime(&c)`가 읽으므로 모든 스레드가 동일한 행 너비를 사용합니다. 나머지 안전성 근거는 명시적입니다. Z 차원은 비활성 상태이며 원시 2차원 실행 형상이 커널과 일치합니다.

:::{tip}
일부러 *단순한* GEMM으로 작성했습니다. 스레드 하나가 원소 하나를 계산하며 공유 메모리 타일링이나 병합 최적화는 없습니다. 실제 운영 GEMM에는 [공유 메모리](../advanced/shared-memory-and-synchronization.md) 장의 기법을 사용합니다. 비동기 조합을 설명하는 이 예제에서는 성능보다 정확성이 우선입니다.
:::

### matvec_naive — 행렬-벡터 곱

```rust
#[kernel]
pub fn matvec_naive(
    _m: u32, n: u32,
    mat: &[f32], vec_in: &[f32],
    mut vec_out: DisjointSlice<f32>,
) {
    let row = thread::index_1d();
    if let Some(out_elem) = vec_out.get_mut(row) {
        let n_sz = n as usize;
        let mut sum = 0.0f32;
        let mut j = 0usize;
        while j < n_sz {
            sum = sum + mat[row.get() * n_sz + j] * vec_in[j];
            j = j + 1;
        }
        *out_elem = sum;
    }
}
```

출력 원소마다 스레드 하나를 사용하고 1차원 인덱싱을 적용합니다. `_m` 매개변수는 사용하지 않지만 BLAS 방식 인터페이스와 호출 규약을 맞춥니다.

### relu — 활성화 함수

```rust
#[kernel]
pub fn relu(input: &[f32], mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = output.get_mut(idx) {
        let val = input[idx.get()];
        *out_elem = if val > 0.0f32 { val } else { 0.0f32 };
    }
}
```

원소별로 `max(0, x)`를 계산합니다. 파이프라인에서 `input`과 `output`은 같은 버퍼를 가리킵니다. 1차원 실행에서 각 스레드가 같은 인덱스를 읽고 쓰므로 유효한 제자리 연산입니다.

### 살펴볼 패턴

세 커널 전반에 반복해서 나타나는 패턴은 다음과 같습니다.

| 패턴                                   | 동작                                                                                               |
| :---------------------------------------- | :----------------------------------------------------------------------------------------------------------|
| `thread::index_1d()` / `index_2d::<S>()`  | 블록/그리드 차원에서 전역 스레드 인덱스 계산                                                |
| `DisjointSlice<f32>`                      | 범위 검사되는 가변 출력. 실행 형상이 고유성 증명을 완성                              |
| `if let Some(elem) = slice.get_mut(idx)`  | 데이터 크기를 벗어나는 스레드를 건너뛰는 범위 검사                                                    |
| `for` 대신 `while` 루프            | 스타일 선택입니다. 범위 기반 `for`도 디바이스에서 동작하지만 `while`은 반복 경계를 명확하게 보여줍니다. |

---

## 호스트와 디바이스 연결

헬퍼 함수 세 개는 원시 CUDA 드라이버 호출을 `DeviceOperation` 값으로 감쌉니다. `with_context`를 사용해 호출 위치가 아니라 실행 시점에 스케줄러의 스트림을 받습니다. 여기서 핵심은 레시피를 먼저 만들고 스케줄러가 나중에 스트림을 선택한다는 점입니다.

### h2d — 호스트에서 디바이스로

```rust
fn h2d(host_data: Vec<f32>) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = host_data.len();
        let num_bytes = n * mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(dptr, host_data.as_ptr(), num_bytes, stream.cu_stream())
                .unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}
```

`host_data` 벡터는 `move`로 캡처됩니다. 스케줄러가 작업을 실행할 때 클로저가 동작하며, 이 시점에는 실제 CUDA 스트림이 준비되어 있습니다. `malloc_async`와 `memcpy_htod_async`는 스트림 순서를 따르므로 할당과 복사는 스케줄러가 선택한 스트림에서 순서대로 처리됩니다.

### zeros — 0으로 초기화한 디바이스 버퍼

```rust
fn zeros(n: usize) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let num_bytes = n * mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memset_d8_async(dptr, 0, num_bytes, stream.cu_stream()).unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}
```

`h2d`와 같은 패턴이지만 `memcpy` 대신 `memset_d8_async`를 사용합니다. GEMM 커널은 `beta = 0.0`을 사용하므로 초기 내용은 중요하지 않지만 0으로 초기화하면 안전합니다.

### d2h — 디바이스에서 호스트로

```rust
fn d2h(dev: DeviceBox<[f32]>) -> impl DeviceOperation<Output = Vec<f32>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = dev.len();
        let num_bytes = n * mem::size_of::<f32>();
        let mut host = vec![0.0f32; n];
        unsafe {
            memcpy_dtoh_async(
                host.as_mut_ptr(), dev.cu_deviceptr(),
                num_bytes, stream.cu_stream(),
            ).unwrap();
        }
        let _ = &dev;
        value(host)
    })
}
```

`let _ = &dev;` 줄은 아무 동작도 하지 않는 것처럼 보이지만 스트림이 동기화될 때까지 클로저 안에서 `dev`를 살아 있게 합니다. 이 줄이 없으면 `memcpy_dtoh_async` 호출 뒤 스트림이 복사를 마치기 전에 `dev`가 Drop될 수 있습니다. 디바이스 측에서 흔히 발생하는 use-after-free입니다.

---

## 파이프라인 구성

각 배치는 조합자로 만든 단일 `DeviceOperation`입니다. 파이프라인을 조립할 때 GPU 작업은 실행되지 않습니다. 이후 수행할 작업을 지연 평가 방식으로 기술합니다. 구성 요소는 다음처럼 연결됩니다.

```{figure} ../async-programming/images/combinator-dataflow.svg
:align: center
:width: 100%

배치 하나의 조합자 파이프라인입니다. `zip!`은 버퍼 세 개를 병렬 할당합니다. `and_then` 체인은 GEMM → MatVec → ReLU → D2H를 순서대로 연결하고 각 단계에서 `value()`로 디바이스 핸들을 전달합니다.
```

### 1단계: 런타임 초기화

```rust
const DIM: usize = 64;
const BLOCK: u32 = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;
```

`init_device_contexts(0, 1)`은 디바이스 0을 기본값으로 설정하고 디바이스 컨텍스트 맵(용량 1)을 초기화합니다. 라운드 로빈 스트림 풀은 처음 사용할 때 지연 생성됩니다. `#[kernel]` 함수에서 컴파일한 내장 모듈은 한 번만 로드되고 타입 지정 모듈 핸들이 공유합니다.

### 2단계: 공유 가중치 업로드

```rust
    let w0_host: Vec<f32> = (0..DIM * DIM)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let w1_host: Vec<f32> = (0..DIM)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();

    let (w0, w1): (Arc<DeviceBox<[f32]>>, Arc<DeviceBox<[f32]>>) =
        zip!(h2d(w0_host).arc(), h2d(w1_host).arc()).await?;
```

독립적인 업로드 두 개를 `zip!`으로 묶어 스트림을 공유할 수 있게 합니다. `.arc()`는 각 결과를 `Arc`로 감쌉니다. 참조 카운트 포인터이므로 복제 비용은 약 1나노초입니다. 배치 네 개, 복제 네 번, 디바이스 복사 0회입니다.

### 3단계: 배치별 파이프라인 구성

핵심 로직입니다. 각 배치마다 네 단계 체인을 구성합니다.

```rust
    use cuda_async::simt::launch::{AsyncKernelLaunchBuilder, OwnedAsyncKernelLaunch};

    let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
        .and_then(move |(input, hidden, output)| {
            // Stage 1: GEMM — hidden = input @ W0
            let func = module.load_function("sgemm_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((
                DIM as u32, DIM as u32, DIM as u32,
                1.0f32,
                input.cu_deviceptr(), input.len() as u64,
                w0.cu_deviceptr(), w0.len() as u64,
                0.0f32,
                hidden.cu_deviceptr(), hidden.len() as u64,
                DIM as u32,  // hidden's row width: the third packet word a
                             // DisjointSlice<_, Runtime2DIndex> takes
            ));
            // SAFETY: packet/config match sgemm_naive (including the width
            // word for `c`), the scheduler uses the module's context, and the
            // owned wrapper retains its allocations.
            let launch = unsafe { builder.finalize_unchecked(gemm_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (input, w0, hidden));
            launch.and_then(move |(_input, _w0, hidden)| {
                value((hidden, output, w1, module))
            })
        })
        .and_then(move |(hidden, output, w1, module)| {
            // Stage 2: MatVec — output = hidden @ W1
            let func = module.load_function("matvec_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((
                DIM as u32, DIM as u32,
                hidden.cu_deviceptr(), hidden.len() as u64,
                w1.cu_deviceptr(), w1.len() as u64,
                output.cu_deviceptr(), output.len() as u64,
            ));
            // SAFETY: packet/config match matvec_naive, the scheduler uses the
            // module's context, and the owned wrapper retains its allocations.
            let launch = unsafe { builder.finalize_unchecked(matvec_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (hidden, w1, output));
            launch.and_then(move |(_hidden, _w1, output)| value((output, module)))
        })
        .and_then(move |(output, module)| {
            // Stage 3: ReLU — result = max(0, output)
            let func = module.load_function("relu").unwrap();
            let relu_out: DeviceBox<[f32]> = output;
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((
                relu_out.cu_deviceptr(), relu_out.len() as u64,
                relu_out.cu_deviceptr(), relu_out.len() as u64,
            ));
            // SAFETY: packet/config match relu, the scheduler uses the module's
            // context, and the owned wrapper retains output.
            let launch = unsafe { builder.finalize_unchecked(relu_cfg) };
            OwnedAsyncKernelLaunch::new(launch, relu_out)
        })
        .and_then(d2h);
```

몇 가지 핵심 부분을 살펴보겠습니다.

**`value()` 바통.** 각 `and_then` 클로저는 이전 단계의 출력을 소비하고 `DeviceOperation`을 반환합니다. 커널 실행은 `()`을 반환하므로 다음 단계에 필요한 디바이스 핸들을 전달하려면 `value()`가 필요합니다. 커널이 실행되고 바통이 다음 단계로 넘어가는 릴레이처럼 생각하면 됩니다.

**타입 주석.** `zip!`과 `and_then` 체인에서 제네릭이 깊게 중첩되면 Rust의 타입 추론 한계를 넘습니다. 클로저 매개변수에 명시적인 타입 주석이 필요합니다.

```rust
.and_then(move |(hidden, output, w1, module): (
    DeviceBox<[f32]>,
    DeviceBox<[f32]>,
    Arc<DeviceBox<[f32]>>,
    Arc<CudaModule>,
)| { ... })
```

이 부분이 사용성 측면의 유일한 거친 부분입니다. `zip!`을 사용하려면 `Zippable` 트레잇도 가져와야 합니다.

**제자리 ReLU.** 3단계는 `relu_out`을 `input`과 `output`으로 모두 커널에 전달합니다. 각 스레드가 같은 인덱스에서 `input[idx]`를 읽고 `output[idx]`를 쓰므로 안전하며 다른 스레드의 쓰기를 읽지 않습니다.

**원시 실행 경계.** 빌더는 아무 작업도 수행하지 않습니다. 원시 패킷과 형상이 실행 가능한 상태가 되는 지점은 `finalize_unchecked`이므로 각 호출에 국소 안전성 근거가 필요합니다. `OwnedAsyncKernelLaunch`는 버퍼 수명을 유지합니다. 커널에 실행 계약이 있으면 `PreparedLaunch<K>`를 사용하는 생성 소유 비동기 메서드를 권장합니다. `c`처럼 행 너비가 런타임에 정해지는 출력은 생성 메서드가 `cuda_host::RowWidthOwned::new(buffer, width)`를 받고 세 번째 패킷 워드를 알아서 구성합니다.

### 4단계: 태스크 생성 및 결과 수집

```rust
    handles.push(tokio::spawn(pipeline.into_future()));
```

`.into_future()`는 지연 실행 `DeviceOperation`을 `DeviceFuture`로 변환합니다. 이때 스케줄링 정책이 스트림을 선택합니다(배치 0 → 스트림 0, 배치 1 → 스트림 1, 라운드 로빈). `tokio::spawn`은 퓨처를 Tokio 런타임에 전달합니다.

첫 폴링에서 파이프라인의 `execute()`가 모든 GPU 작업을 할당된 스트림에 제출하고 `cuLaunchHostFunc` 콜백을 등록합니다. 퓨처는 `Poll::Pending`을 반환합니다. GPU 작업이 끝나면 콜백이 태스크를 깨웁니다. 호스트 스레드가 바쁘게 대기하지 않습니다.

```rust
    for (i, handle) in handles.into_iter().enumerate() {
        let result: Vec<f32> = handle.await.expect("Tokio task panicked")?;
        let all_non_negative = result.iter().all(|&v| v >= 0.0);
        println!("Batch {}: {} elements, first 8 = {:?}{}",
            i, result.len(), &result[..8.min(result.len())],
            if all_non_negative { " [ReLU OK]" } else { " [ReLU FAILED]" }
        );
    }
```

ReLU 검사는 딥러닝 검증이 아니라 활성화 함수가 실행됐는지만 확인합니다. 모든 출력은 0 이상이어야 합니다. `[ReLU FAILED]`가 보이면 앞 단계에 심각한 문제가 있는 것입니다.

---

## 빌드, 실행, 검증

```bash
cargo oxide run async_mlp
```

예상 출력:

```text
=== Async MLP Pipeline ===

Allocating model weights...
  W0: 64x64 on device (Arc refcount=1)
  W1: 64 on device (Arc refcount=1)

Launched 4 batches concurrently, awaiting results...

Batch 0: 64 elements, first 8 = [0.0020799995, 0.0, ...] [ReLU OK]
Batch 1: 64 elements, first 8 = [0.0, 0.0, ...] [ReLU OK]
Batch 2: 64 elements, first 8 = [0.0, 0.00108, ...] [ReLU OK]
Batch 3: 64 elements, first 8 = [0.00244, 0.0025, ...] [ReLU OK]

SUCCESS: All batches completed.
```

Arc 참조 카운트는 1에서 시작합니다(`w0`, `w1` 각각 참조 하나). 배치를 처리하는 동안 원본과 복제본 네 개를 합쳐 5까지 올랐다가 배치가 끝나면 다시 줄어듭니다. 배치를 더 추가하면 참조 카운트도 비례해 증가하지만 복사나 재할당은 없습니다.

### Nsight Systems로 프로파일링

스트림 중첩을 확인하려면 다음을 실행하세요.

```bash
nsys profile --trace=cuda cargo oxide run async_mlp
nsys-ui report.nsys-rep
```

타임라인에서 스트림마다 커널 블록 행이 하나씩, 총 네 행이 병렬로 표시되는지 확인하세요. 커널이 한 스트림에서 직렬 실행된다면 라운드 로빈 스케줄러를 사용하지 않는 것입니다(`init_device_contexts`를 호출하지 않았거나 스트림을 하나만 설정했을 가능성이 큽니다).

---

## 실제 사례와 닮은 점

64×64 행렬, 임의 가중치, 커널 세 개를 사용하는 작은 예제입니다. 하지만 *구조*는 실제 운영 GPU 파이프라인과 같습니다.

| 운영 환경의 패턴                                | async_mlp에서의 대응                                |
| :------------------------------------------------ | :-------------------------------------------------- |
| 모델 가중치를 한 번 로드해 여러 요청에서 공유 | `zip!(h2d(w0).arc(), h2d(w1).arc())`                |
| 요청별 추론 파이프라인                    | `and_then` 체인: GEMM → MatVec → ReLU → D2H        |
| 동시 요청 처리                     | 배치별 `tokio::spawn(pipeline.into_future())`    |
| 스트림 기반 GPU 스케줄링                       | 라운드 로빈 `init_device_contexts(0, 1)`            |
| 논블로킹 호스트                                 | `.sync()` 대신 join 핸들에 `.await`             |

행렬을 64에서 4096으로 키우고 단순 커널을 타일 공유 메모리 버전([공유 메모리](../advanced/shared-memory-and-synchronization.md) 참조)으로 바꾸고 계층을 추가하면 실제 추론 서버의 뼈대가 됩니다.

---

## 예제 확장

더 확장해 볼 아이디어입니다.

**소프트맥스 계층 추가.** 출력 벡터 64개 원소에 대해 [수치적으로 안정적인 소프트맥스](https://en.wikipedia.org/wiki/Softmax_function)를 계산하는 `#[kernel]`을 작성하고 ReLU 뒤에 `.and_then`으로 연결하세요.

**더 큰 차원에서 프로파일링.** `DIM`을 512 또는 1024로 바꾸고 GEMM이 타임라인 대부분을 차지하는지 살펴보세요. 그런 다음 `sgemm_naive`를 `SharedArray` 기반 타일 버전으로 바꾸어 속도 향상을 관찰하세요.

**오류 복구 추가.** 커널 실행 클로저의 `.unwrap()`을 적절한 `Result` 전파로 바꾸세요. GPU 오류와 태스크 패닉을 따로 처리하려면 {ref}`동시 실행 <concurrent-error-handling>` 장의 세 갈래 `match` 패턴을 사용하세요.

**멀티 디바이스.** GPU 두 개를 관리하도록 `init_device_contexts(0, 2)`를 전달하세요. 짝수 배치는 GPU 0, 홀수 배치는 GPU 1로 보내는 배치 분할기를 구성하세요.

:::{seealso}
- [DeviceOperation 모델](../async-programming/the-device-operation-model.md) — 지연 실행 GPU 그래프의 동작 방식
- [조합자와 구성](../async-programming/combinators-and-composition.md) — `and_then`, `zip!`, `value()`, `.arc()` 상세 설명
- [스케줄링과 스트림](../async-programming/scheduling-and-streams.md) — 라운드 로빈, 스트림 풀, 스케줄링 정책
- [동시 실행](../async-programming/concurrent-execution.md) — 이 장의 기반이 되는 이론
:::

# DeviceOperation 모델

[GPU 프로그램 작성](https://nvidia.github.io/cuda-rust/cuda-oxide/gpu-programming/launching-kernels.html) 장에서 타입 지정 동기 실행은 명시한 스트림에 작업을 넣고, 비동기 실행은 스트림 선택을 미루는 지연 핸들을 반환한다는 점을 살펴봤습니다. 이 장에서는 그 핸들의 기반인 **`DeviceOperation`** 트레이트와 GPU가 *무엇을* 할지와 *어느 스트림에서* 실행할지를 분리하는 것이 조합형 비동기 GPU 프로그래밍의 토대인 이유를 설명합니다.

:::{seealso}
기반이 되는 CUDA 스트림 및 이벤트 모델은 [CUDA 프로그래밍 가이드 — 비동기 동시 실행](https://docs.nvidia.com/cuda/cuda-programming-guide/#asynchronous-concurrent-execution)을 참고하세요.
:::

## 지연 연산을 사용하는 이유

CUDA C++에서는 `cudaStream_t`를 여러 개 만들고 커널 실행과 메모리 복사를 스트림별로 직접 배치해 동시성을 구성합니다. 호출하는 곳마다 스트림을 결정해야 하므로 GPU 작업의 *정의*와 *스케줄링 결정*이 결합되어 나중에 작업을 조합하거나 재배치하기 어렵습니다.

cuda-oxide의 `DeviceOperation`은 스트림에 묶이지 않은 GPU 작업을 표현합니다. `and_then`, `zip!` 같은 조합기로 연결하고, 함수 사이에 전달하거나 컬렉션에 저장한 뒤 마지막 순간에 실행 정책을 정할 수 있습니다. Rust `Iterator`처럼 지연적으로 파이프라인을 만들고 실행 지점에서 동작시킵니다.

| 방식 | 스트림 선택 주체 | 조합 가능 여부 |
|:--|:--|:--|
| 타입 지정 동기 실행 | 호출자 | 즉시 제출하므로 조합할 수 없음 |
| 타입 지정 비동기 실행 | 스케줄링 정책 | 가능; `DeviceOperation` 반환 |

```{figure} images/device-operation-lifecycle.svg
:align: center
:width: 100%

DeviceOperation 수명 주기. 1단계: 타입 지정 비동기 메서드가 GPU 작업 없이 지연 레시피를 구성합니다. 2단계: 스케줄러가 풀에서 스트림을 선택합니다. 3단계: `execute()`가 작업과 `cuLaunchHostFunc` 콜백을 제출합니다. 4단계: 콜백이 실행되어 비동기 런타임을 깨우고 결과를 전달합니다. 아래는 간단한 `.sync()`부터 수동 제어 수준이 높은 `async_on`까지 네 가지 실행 방법입니다.
```

## 레시피와 주방

`DeviceOperation`은 **레시피 카드**와 같습니다. 재료를 무엇과 섞고 얼마나 조리할지는 적혀 있지만 *어느 주방*에서 만들지는 정하지 않습니다. 카드를 어느 주방에나 전달하고, 복사하거나 여러 코스를 묶거나 나중을 위해 보관할 수 있습니다. 누군가 주방에서 지침을 실행할 때 조리가 시작됩니다.

- **레시피**는 지연 GPU 작업 설명인 `DeviceOperation`입니다.
- **주방**은 작업이 순서대로 실행되는 CUDA 스트림입니다.
- **주방장**은 레시피를 어느 주방에 맡길지 정하는 `SchedulingPolicy`입니다.
- **요리**는 완료 후 결과를 나타내는 `Output`입니다.

이 분리 덕분에 어느 스트림에서 실행할지 모르는 함수도 “데이터 업로드, GEMM, ReLU” 레시피를 반환할 수 있습니다. 호출자는 단계를 더 연결하거나 특정 스트림에서 실행하거나 정책에 맡길 수 있습니다.

## 첫 비동기 실행

`DeviceOperation`을 만드는 가장 간단한 방법은 생성된 `{kernel}_async` 메서드입니다. 동기 메서드와 비슷하지만 스트림 인자를 받지 않고 바로 실행하는 대신 레시피를 반환합니다.

```rust
use cuda_async::simt::device_context::init_device_contexts;
use cuda_core::simt::LaunchConfig;

// 한 번만 설정: 스케줄링용 스트림 풀 생성
init_device_contexts(0, 1)?;

// 아직 GPU 작업은 하지 않고 레시피만 만듭니다.
let module = kernels::load_async(0)?;
// SAFETY: 1D 실행이며 버퍼는 각각 1024개 원소를 담고,
// 모듈과 스케줄러가 모두 장치 0의 컨텍스트를 사용합니다.
let op = unsafe {
    module.vecadd_async(
        LaunchConfig::for_num_elems(1024),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )
}?;

// 이제 스트림을 선택하고 실행한 뒤 결과를 기다립니다.
op.sync()?;
```

`op`을 만들 때 GPU는 아무 작업도 하지 않았습니다. `AsyncKernelLaunch` 값에 호출할 함수, 인자, 그리드 설정을 기록했지만 스트림에는 접근하지 않았습니다. `unsafe`는 지연 구성 때문이 아니라 증명되지 않은 원시 `LaunchConfig`를 받아들이기 때문입니다. 실행 계약이 있는 커널은 검사된 `PreparedLaunch<K>`를 쓰므로 안전한 생성 비동기 메서드를 사용할 수 있습니다.

```text
원시 설정 -> unsafe 비동기 레시피
준비된 K -> K에 안전하게 타입 지정된 비동기 레시피
```

`.sync()`를 호출하면 정책이 풀에서 스트림을 정하고 커널을 제출한 뒤 완료까지 블로킹합니다.

## `DeviceOperation`을 구성하는 요소

`DeviceOperation`은 GPU 작업을 설명하는 타입이 구현할 수 있는 트레이트입니다. 필수 메서드 하나와 연관 타입 하나를 가집니다.

```rust
pub trait DeviceOperation: Send + Sized + IntoFuture {
    type Output: Send;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<Self::Output, DeviceError>;
}
```

**`Output`**은 작업 완료 시 얻는 Rust 값입니다. 커널 실행은 디바이스 메모리에 부수 효과를 남기므로 보통 `()`를 반환하고, 디바이스-호스트 복사는 `Vec<f32>`, 메모리 할당은 포인터를 소유하는 `DeviceBox<[f32]>`를 반환할 수 있습니다.

**`execute`**가 실제 GPU 작업을 수행합니다. 배정된 스트림을 포함하는 `ExecutionContext`를 받아 작업을 제출합니다. 반환 뒤에도 GPU 작업이 진행 중일 수 있으므로 결과를 읽기 전에 동기화할 책임이 있어 unsafe입니다. `Send`는 Tokio 작업 등 스레드 간 이동을 허용하고, `IntoFuture`는 `.await`를 가능하게 합니다.

직접 구현하는 경우는 드뭅니다. 주요 구현 타입은 다음과 같습니다.

- `AsyncKernelLaunch` — 타입 지정 비동기 실행 메서드가 만들며 커널을 실행합니다.
- `Value<T>` — 호스트 값을 감싸며 GPU 작업 없이 `T`를 반환합니다.
- `AndThen` — A를 실행하고 결과를 B에 전달합니다.
- `Zip` — 두 작업을 실행해 결과 튜플을 반환합니다.
- `StreamOperation` — 스트림이 정해진 뒤 작업을 구성합니다.

## `ExecutionContext` — 스트림이 있는 곳

레시피를 실행하려면 어느 주방인지 알아야 합니다. `ExecutionContext`가 GPU, 스트림, CUDA 컨텍스트 정보를 담습니다.

```rust
pub struct ExecutionContext {
    device: usize,                  // GPU 식별자
    cuda_stream: Arc<CudaStream>,   // 스트림
    cuda_context: Arc<CudaContext>, // CUDA 컨텍스트
}
```

작업 자체는 스트림을 만들지 않습니다. [스케줄링과 스트림](scheduling-and-streams.md) 장에서 다루는 정책이 컨텍스트를 만들고 `execute`에 전달합니다. 즉 작업은 *무엇을*, 컨텍스트는 *어디서* 실행할지를 나타냅니다. `execute` 구현에서는 `ctx.get_cuda_stream()`과 `ctx.get_cuda_context()`로 접근합니다.

## 레시피 실행 방법 네 가지

### `.sync()` — 블로킹 실행

일반적인 동기 실행입니다. 스케줄링 정책이 스트림을 골라 작업을 제출하고 결과가 나올 때까지 호출 스레드를 블로킹합니다. 비동기 런타임 밖의 간단한 코드에 적합합니다.

```rust
let result = operation.sync()?;
```

### `.await` — 양보 후 재개

`DeviceOperation`은 `IntoFuture`를 구현하므로 비동기 함수에서 기다릴 수 있습니다. 작업이 진행되는 동안 현재 Tokio 작업을 양보하므로 런타임의 다른 작업이 실행됩니다.

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;
    let result = operation.await?;
    Ok(())
}
```

### `.sync_on(&stream)` — 스트림 직접 선택

호출자가 스트림을 정해 작업을 동기적으로 실행합니다. 관련 작업의 순서를 명시해야 하며 호출 스레드는 완료될 때까지 블로킹됩니다.

```rust
let result = operation.sync_on(&stream)?;
```

### `unsafe async_on(&stream)` — 비동기 제출

지정한 스트림에 작업을 비동기 제출하고 완료는 기다리지 않습니다. 반환 뒤에도 메모리 접근이 진행될 수 있어 호출자는 버퍼 수명과 이후 동기화를 보장해야 합니다.

```rust
unsafe { operation.async_on(&stream)?; }
```

## `value()`로 호스트 데이터 전달

이미 가진 호스트 값을 작업 그래프 안에 넣을 때 `value(x)`를 사용합니다. GPU 작업을 하지 않고 값을 즉시 반환하므로, GPU 단계 결과와 기존 값을 한 체인에서 조합할 수 있습니다.

```rust
let pipeline = value(host_metadata)
    .and_then(|metadata| upload_and_process(metadata));
```

## `with_context`로 스트림에 접근

CUDA 드라이버 함수를 직접 호출해야 한다면 `with_context`를 사용합니다. 작업을 만들 때는 스트림이 아직 정해지지 않았으므로 실제 스트림에 접근하는 클로저 실행은 작업이 시작될 때까지 미뤄집니다.

```rust
let copy = with_context(move |ctx| {
    let stream = ctx.get_cuda_stream();
    // 실제 선택된 스트림에서 드라이버 작업 구성
    submit_copy(stream, data)
});
```

이 방식은 원시 CUDA 호출을 감싼 `h2d`, `d2h`, `zeros` 같은 도우미를 구현할 때 유용합니다. `and_then_with_context`는 체인 중간에서 직전 결과와 컨텍스트가 모두 필요할 때 사용합니다.

## GPU 완료를 Rust에 알리는 방법

GPU는 호스트와 별도로 작업하므로 퓨처가 바쁜 대기 없이 완료를 알아야 합니다. cuda-oxide는 CUDA 스트림에 `cuLaunchHostFunc` 콜백을 등록합니다. 실행이 끝나면 콜백이 비동기 퓨처의 작업 깨우기(waker)를 호출하고, 런타임은 작업을 다시 폴링해 결과를 전달합니다. GPU가 도는 동안 호스트 스레드는 다른 작업을 처리할 수 있습니다.

:::{seealso}
조합기 상세는 [조합기와 연산 구성](combinators-and-composition.md), 스트림 선택은 [스케줄링과 스트림](scheduling-and-streams.md), 독립 파이프라인 동시 실행은 [동시 실행](concurrent-execution.md)을 참고하세요.
:::

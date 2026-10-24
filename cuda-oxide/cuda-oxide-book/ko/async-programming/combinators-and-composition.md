# 조합기와 연산 구성

커널 실행 한 번으로 끝나는 GPU 작업은 드뭅니다. 실제 작업에서는 입력 업로드, 행렬 곱, 활성화 함수 적용, 결과를 호스트로 복사하는 네 작업을 정확한 순서로 실행할 수 있습니다. 이 장에서는 cuda-oxide 조합기로 지연성과 스트림 독립성을 유지하는 파이프라인을 만드는 방법을 설명합니다.

:::{seealso}
조합기의 기반이 되는 트레이트와 실행 메서드는 [DeviceOperation 모델](the-device-operation-model)을 참고하세요.
:::

## 문제: 여러 단계의 GPU 작업

간단한 신경망 순방향 패스를 생각해 봅시다.

1. 호스트에서 디바이스로 입력 배치를 업로드합니다.
2. **GEMM**으로 입력과 가중치 행렬 W0을 곱합니다.
3. 결과에 **ReLU** 활성화 함수를 적용합니다.
4. 출력을 호스트로 내려받습니다.

각 단계는 앞 단계가 끝나야 합니다. 동기 방식에서는 `.sync()`를 네 번 호출할 수 있습니다.

```rust
let input  = h2d(batch_data).sync()?;       // 업로드
let hidden = launch_gemm(input, w0).sync()?; // GEMM
let output = launch_relu(hidden).sync()?;    // ReLU
let result = d2h(output).sync()?;            // 다운로드
```

하지만 각 `.sync()`가 스케줄링 정책을 거쳐 매번 다른 스트림을 선택할 수 있습니다. 스트림 사이에는 순서 보장이 없으므로 다른 스트림의 업로드가 끝나기 전에 GEMM이 시작될 수 있습니다. 같은 스트림이 선택되더라도 `.sync()`마다 호스트 스레드가 블로킹됩니다. 매번 작업 하나를 제출하고 기다리는 왕복 비용을 지불하는 셈입니다.

원하는 것은 전체 실행 계획을 한 번에 작성해 스트림 하나에 넘기는 것입니다.

## `and_then` — “이 작업이 끝나면 다음 작업 실행”

`and_then`은 두 작업을 연결합니다. 첫 번째 작업을 실행하고 결과를 클로저에 전달하면 클로저가 다음 작업을 반환합니다. 두 작업은 **같은 스트림**에서 실행되며 CUDA 스트림의 순서 보장에 따라 두 번째 작업은 첫 번째 결과를 볼 수 있습니다.

```rust
let pipeline = h2d(batch_data)
    .and_then(|input| launch_gemm(input, w0))
    .and_then(|hidden| launch_relu(hidden))
    .and_then(|output| d2h(output));

let result: Vec<f32> = pipeline.sync()?;
```

코드를 작성할 때 GPU 작업은 실행되지 않습니다. 전체 체인을 하나의 `DeviceOperation`으로 표현합니다. `.sync()` 시 스케줄러가 스트림 하나를 고르고 체인을 위에서 아래로 실행합니다.

### 체인에서 데이터 전달

각 `and_then` 클로저는 앞 단계의 `Output`을 받아 다음 단계의 `DeviceOperation`을 반환합니다.

```text
h2d(batch)         → DeviceBox<[f32]>
    │
    └─ and_then ─► launch_gemm(input, w0)   → DeviceBox<[f32]>
                        │
                        └─ and_then ─► launch_relu(hidden)   → DeviceBox<[f32]>
                                            │
                                            └─ and_then ─► d2h(output)   → Vec<f32>
```

체인 전체 타입은 컴파일러가 추론하므로 중간 타입을 직접 선언할 필요가 없습니다.

### `value()`로 추가 값 전달

커널 실행은 반환값 대신 디바이스 메모리에 부수 효과를 남기므로 `()`를 반환합니다. 다음 단계에서 버퍼 핸들, 모듈 참조, 메타데이터가 필요하다면 `value()`로 묶어 전달합니다.

```rust
let pipeline = launch_gemm(input, hidden)
    .and_then(move |()| {
        // 커널은 ()를 반환했지만 바깥 범위의 hidden과 module은 남아 있습니다.
        value((hidden, module))
    })
    .and_then(move |(hidden, module)| {
        launch_relu(hidden, module)
    });
```

`move`는 중요합니다. 클로저가 필요한 값을 소유권으로 캡처하고 실행할 때 소비한 뒤 `value()`로 전달합니다. Rust 소유권 규칙은 버퍼가 한 번에 정확히 하나의 단계에서 사용되도록 하고 매달린 참조를 방지합니다.

### `and_then_with_context` — 스트림이 필요한 경우

단계 사이에서 비동기 메모리 복사, 이벤트 기록, 동기화 같은 원시 CUDA 연산을 해야 할 때가 있습니다. 일반 `and_then` 클로저에는 `CUstream` 핸들이 없지만 `and_then_with_context`는 이전 결과와 `ExecutionContext`를 함께 제공합니다.

```rust
let pipeline = launch_kernel(input)
    .and_then_with_context(|ctx, gpu_result| {
        let stream = ctx.get_cuda_stream();
        copy_result_to_staging(stream, gpu_result)
    });
```

드물게 사용하세요. 대부분의 파이프라인은 `and_then`과 내부적으로 `with_context`를 사용하는 `h2d`, `d2h`, `zeros` 도우미만으로 구성할 수 있습니다.

## `zip!` — 독립 작업 묶기

순방향 패스를 시작하기 전에 입력, 은닉층 스크래치, 출력용 버퍼 세 개를 할당해야 한다고 합시다. 서로 의존하지 않는 작업이지만 결과는 모두 필요합니다. 세 작업을 전부 `and_then`으로 연결하면 결과를 중첩 클로저에서 전달해야 합니다.

```rust
// 동작은 하지만 읽기 어렵습니다.
let pipeline = h2d(batch_data)
    .and_then(|input| {
        zeros(DIM * DIM).and_then(move |hidden| {
            zeros(DIM).and_then(move |output| {
                value((input, hidden, output))
            })
        })
    });
```

`zip!`은 독립 작업을 하나로 묶고 각 결과를 튜플로 반환합니다.

```rust
use cuda_async::zip;

let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM));
// pipeline: impl DeviceOperation<Output = (DeviceBox, DeviceBox, DeviceBox)>

let (input, hidden, output) = pipeline.sync()?;
```

`zip!`은 인자를 두 개 또는 세 개 받으며 같은 스트림에서 순서대로 실행합니다. 결과들을 튜플로 수집합니다. 여기서 `zip`은 독립 결과를 튜플로 결합한다는 뜻이지 병렬 실행을 의미하지 않습니다. 스트림 간 동시 실행은 [동시 실행](concurrent-execution.md)을 참고하세요.

### `zip!`과 `and_then` 조합

`zip!`으로 독립 설정을 처리하고 `and_then`으로 의존 파이프라인을 연결합니다.

```rust
let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
    .and_then(|(input, hidden, output)| launch_gemm(input, hidden, w0))
    .and_then(|hidden| launch_relu(hidden))
    .and_then(|result| d2h(result));
```

버퍼 세 개를 준비하고 GEMM, ReLU, 다운로드를 실행한다는 흐름이 드러납니다. `.sync()` 또는 `.await`할 때까지 실행하지 않는 `DeviceOperation` 하나입니다.

## `.arc()` — 파이프라인 사이에서 결과 공유

배치 처리에서 모델 가중치를 한 번 로드해 여러 순방향 패스가 공유할 수 있습니다. 하지만 각 `DeviceOperation`은 출력을 값으로 소비하므로 같은 `DeviceBox`를 네 클로저에 이동할 수 없습니다. `.arc()`는 결과를 `Arc<T>`로 감싸 저렴하게 복제하도록 합니다.

```rust
let (w0, w1) = zip!(
    h2d(w0_host).arc(),
    h2d(w1_host).arc()
).await?;

// w0: Arc<DeviceBox<[f32]>>
for batch in batches {
    let w0 = w0.clone();
    let w1 = w1.clone();
    tokio::spawn(build_forward_pass(batch, w0, w1).into_future());
}
```

각 배치 파이프라인은 저렴한 `Arc::clone`으로 가중치 버퍼를 공유합니다. 어느 배치든 사용 중이면 가중치가 살아 있으며 참조 카운팅이 정리를 맡습니다.

## `unzip!` — 쌍 결과 나누기

연산이 튜플을 만들지만 각 원소를 서로 다른 후속 체인에 전달하고 싶을 수 있습니다. `unzip!`은 쌍을 만드는 연산을 실행을 공유하는 두 작업으로 나눕니다. 원본은 최대 한 번만 실행됩니다.

```rust
use cuda_async::unzip;

let pair_op = zip!(allocate_a(), allocate_b());
let (op_a, op_b) = unzip!(pair_op);

let result_a = op_a.and_then(|a| process_a(a));
let result_b = op_b.and_then(|b| process_b(b));
```

내부적으로 `unzip!`은 메모이제이션 노드를 만듭니다. 먼저 실행되는 분기가 원본을 실행하고 두 번째 분기는 캐시된 결과를 읽습니다. 공통 설정 이후 파이프라인을 갈라야 할 때 유용합니다.

## 모두 조합하기: MLP 순방향 패스

```{figure} images/combinator-dataflow.svg
:align: center
:width: 100%

조합기로 만든 MLP 순방향 패스 데이터 흐름입니다. `zip!`이 시작 부분에서 독립 할당 세 개를 묶고, 네 개의 `and_then` 단계가 GEMM, MatVec, ReLU, D2H를 차례로 연결합니다. `value()` 튜플로 단계 간 타입을 전달합니다. `.sync()` 또는 `.await` 전까지는 하나의 `DeviceOperation` 전체가 실행되지 않습니다.
```

다음은 모든 조합기를 함께 사용하는 `async_mlp` 예제의 단순화 버전입니다. 배치 하나의 순방향 패스 전체를 표현하는 단일 `DeviceOperation`을 반환합니다.

```rust
use cuda_async::simt::launch::{AsyncKernelLaunchBuilder, OwnedAsyncKernelLaunch};

fn forward_pass(
    batch: Vec<f32>,
    w0: Arc<DeviceBox<[f32]>>,
    w1: Arc<DeviceBox<[f32]>>,
    module: Arc<CudaModule>,
) -> impl DeviceOperation<Output = Vec<f32>> {
    // 설정: 독립적인 입력과 스크래치 버퍼 할당
    zip!(h2d(batch), zeros(DIM * DIM), zeros(DIM))
        // 1단계: GEMM — hidden = input × W0
        .and_then(move |(input, hidden, output)| {
            let func = module.load_function("sgemm_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((DIM as u32, DIM as u32, DIM as u32,
                               1.0f32,
                               input.cu_deviceptr(), input.len() as u64,
                               w0.cu_deviceptr(), w0.len() as u64,
                               0.0f32,
                               hidden.cu_deviceptr(), hidden.len() as u64));
            // SAFETY: 인자와 실행 설정이 sgemm_naive에 맞고,
            // 스케줄러 컨텍스트가 모듈과 같으며 owned wrapper가 할당을 유지합니다.
            let launch = unsafe { builder.finalize_unchecked(gemm_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (input, w0, hidden));
            launch.and_then(move |(_input, _w0, hidden)| {
                value((hidden, output, w1, module))
            })
        })
        // 2단계: MatVec — output = hidden × W1
        .and_then(move |(hidden, output, w1, module)| {
            let func = module.load_function("matvec_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((DIM as u32, DIM as u32,
                               hidden.cu_deviceptr(), hidden.len() as u64,
                               w1.cu_deviceptr(), w1.len() as u64,
                               output.cu_deviceptr(), output.len() as u64));
            // SAFETY: 인자와 설정이 matvec_naive에 맞고 모듈 컨텍스트를
            // 사용하며 owned wrapper가 할당을 보존합니다.
            let launch = unsafe { builder.finalize_unchecked(matvec_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (hidden, w1, output));
            launch.and_then(move |(_hidden, _w1, output)| value((output, module)))
        })
        // 3단계: ReLU — result = max(0, output)
        .and_then(move |(output, module)| {
            let func = module.load_function("relu").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((output.cu_deviceptr(), output.len() as u64,
                               output.cu_deviceptr(), output.len() as u64));
            // SAFETY: 인자와 설정이 relu에 맞고 모듈 컨텍스트를 사용하며
            // owned wrapper가 출력 버퍼를 유지합니다.
            let launch = unsafe { builder.finalize_unchecked(relu_cfg) };
            OwnedAsyncKernelLaunch::new(launch, output)
        })
        // 4단계: 결과를 호스트로 다운로드
        .and_then(d2h)
}
```

데이터는 다음 순서로 전달됩니다.

1. `zip!`이 `DeviceBox` 핸들 세 개를 만듭니다.
2. 첫 `and_then`이 이를 받아 GEMM을 실행하고 `value()` 튜플로 필요한 버퍼를 전달합니다.
3. 두 번째 단계가 튜플을 풀어 MatVec를 실행하고 다음 단계에 필요한 값만 전달합니다.
4. 세 번째 단계가 제자리에서 ReLU를 실행합니다.
5. 마지막 단계의 `d2h`가 결과를 호스트로 복사합니다.

함수는 `impl DeviceOperation<Output = Vec<f32>>`를 반환하며 이 시점에는 실행된 것이 없습니다. `.sync()` 또는 `.await`하거나 `tokio::spawn`에 전달할 수 있고, 스트림 선택은 스케줄러가 합니다.

`AsyncKernelLaunchBuilder`는 실행 전까지 부수 효과가 없습니다. `finalize_unchecked(raw_config)`에서만 unsafe 경계를 넘어 실행 가능한 작업을 만듭니다. `OwnedAsyncKernelLaunch`는 커널 실행이 버퍼를 반환할 때까지 각 단계의 버퍼를 유지합니다. 애플리케이션은 가능하면 `PreparedLaunch<K>`를 사용하는 생성된 owned async 메서드를 사용하세요. 위 저수준 방식은 모듈을 이름으로 로드해 생성 계약을 사용할 수 없을 때 유용합니다.

## 빠른 참조

| 조합기 | 동작 | 사용 시점 |
|:--|:--|:--|
| `.and_then(f)` | 같은 스트림에서 작업 후 `f(result)` 실행 | 의존 단계 연결 |
| `.and_then_with_context(f)` | 컨텍스트도 전달하는 `and_then` | 원시 스트림이 필요할 때 |
| `.apply(f)` | `.and_then`의 별칭 | 코드 문맥에 더 읽기 좋은 쪽 |
| `zip!(a, b)` | 두 작업 실행 후 `(A, B)` 반환 | 독립 설정 작업 |
| `zip!(a, b, c)` | 세 작업 실행 후 `(A, B, C)` 반환 | 독립 작업 세 개 |
| `.arc()` | 출력을 `Arc<T>`로 감쌈 | 파이프라인 사이 결과 공유 |
| `unzip!(op)` | 튜플 결과 작업을 두 작업으로 분리 | 갈라지는 후속 체인 |
| `value(x)` | `x`를 no-op 작업으로 감쌈 | 호스트 데이터를 파이프라인에 전달 |
| `with_context(f)` | 스트림이 정해질 때 구성 지연 | 원시 CUDA 드라이버 호출 래핑 |

:::{seealso}
여러 파이프라인을 `tokio::spawn`으로 동시에 실행하는 방법은 [동시 실행](concurrent-execution.md), 스케줄링 정책이 CUDA 스트림에 파이프라인을 분배하는 방식은 [스케줄링과 스트림](scheduling-and-streams.md)을 참고하세요.
:::

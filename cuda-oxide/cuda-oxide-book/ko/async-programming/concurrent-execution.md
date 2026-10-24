# 동시 실행

`DeviceOperation`으로 GPU 작업을 표현하고, `and_then`과 `zip!`으로 여러 단계의 파이프라인을 조합하며, 스케줄링 정책이 스트림을 배정하도록 하는 방법을 살펴보았습니다. 이제 여러 파이프라인을 **동시에** 실행해 보겠습니다. 이 장에서는 `async_mlp` 예제를 처음부터 끝까지 따라가며 Tokio 작업, 라운드 로빈 스케줄러, Rust 소유권 모델이 한 GPU에서 네 배치를 동시에 처리하는 방식을 설명합니다.

:::{seealso}
멀티 GPU 컨텍스트, 피어 액세스, 장치 간 메모리에 관한 CUDA 규칙은 [CUDA 프로그래밍 가이드 — 다중 장치 시스템](https://docs.nvidia.com/cuda/cuda-programming-guide/#multi-device-system)을 참고하세요.
:::

## 실행 시나리오

3계층 MLP 순방향 패스를 실행합니다. 연산은 **GEMM**(행렬 곱), **MatVec**(행렬-벡터 곱), **ReLU**(활성화 함수)입니다. GPU에 가중치 행렬 두 개가 올라가 있고, 처리할 입력 배치는 네 개입니다. 각 배치는 서로 독립적이지만 모든 배치가 같은 가중치를 공유합니다.

```{figure} images/concurrent-batches.svg
:align: center
:width: 100%

네 개의 MLP 순방향 패스를 동시에 실행합니다. 위쪽은 `Arc<DeviceBox>`로 가중치를 한 번 올린 뒤 각 배치에서 저렴하게 복제하는 모습입니다. 가운데 GPU 타임라인에서는 라운드 로빈 방식으로 배치를 네 스트림에 분배하고 파이프라인 실행을 겹칩니다. 아래쪽은 순차 처리 시간이 단일 배치의 약 4배인 반면, GPU SM 여유가 있을 때 동시 처리는 약 1.3배임을 보여줍니다.
```

순차 방식은 배치를 하나씩 처리합니다.

```text
Batch 0:  ████ GEMM ████ MatVec ██ ReLU █ D2H █
                                                  Batch 1:  ████ GEMM ████ ...
```

동시 방식은 여러 스트림에서 작업을 겹칩니다.

```text
Stream 0:  ████ GEMM ████ MatVec ██ ReLU █ D2H █
Stream 1:    ████ GEMM ████ MatVec ██ ReLU █ D2H █
Stream 2:      ████ GEMM ████ MatVec ██ ReLU █ D2H █
Stream 3:        ████ GEMM ████ MatVec ██ ReLU █ D2H █
```

GPU에 여러 커널을 동시에 실행할 만큼 SM 여유가 있으면 겹쳐 실행하는 방식이 훨씬 빨리 끝납니다. 각 파이프라인은 하나의 스트림에서 단일 `and_then` 체인으로 구성되므로 배치 내부 단계의 순서는 그대로 유지됩니다. 스트림 간 동기화도 필요하지 않습니다.

## 1단계: 런타임 초기화

먼저 `init_device_contexts`를 호출합니다. 이 함수는 CUDA 컨텍스트를 만들고 기본 네 개 스트림을 사용하는 라운드 로빈 스케줄링 정책을 설정하며, `.sync()`와 `.await`에서 사용할 스레드 로컬 상태를 준비합니다.

```rust
use cuda_async::simt::device_context::init_device_contexts;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;
```

임베디드 모듈은 스레드 로컬 비동기 컨텍스트를 통해 로드합니다. 타입이 지정된 모듈 핸들은 각 배치 파이프라인에 저렴하게 복제할 수 있습니다.

## 2단계: 공유 가중치 업로드

모델에는 두 개의 가중치 행렬이 있습니다. W0의 크기는 DIM × DIM이고 W1의 크기는 DIM입니다. 네 번의 순방향 패스가 모두 끝날 때까지 두 행렬은 장치에 있어야 합니다. 여기서 `zip!`과 `.arc()`를 활용할 수 있습니다.

```rust
    let w0_host: Vec<f32> = (0..DIM * DIM)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let w1_host: Vec<f32> = (0..DIM)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();

    let (w0, w1): (Arc<DeviceBox<[f32]>>, Arc<DeviceBox<[f32]>>) = zip!(
        h2d(w0_host).arc(),
        h2d(w1_host).arc()
    ).await?;
```

`zip!`은 서로 독립적인 두 H2D 전송을 하나의 연산으로 묶습니다. `.arc()`는 결과를 각각 `Arc`로 감싸 공유할 수 있게 합니다. `.await`는 결합한 연산을 스트림 풀의 스트림 하나에 예약하고 완료를 기다립니다.

이후 `w0`와 `w1`의 타입은 `Arc<DeviceBox<[f32]>>`입니다. 복제 비용이 낮고 안전하게 공유할 수 있으며, 장치 메모리에 유지됩니다.

## 3단계: 배치 파이프라인 생성 및 실행

이제 배치별로 지연 파이프라인을 만듭니다. 아직 GPU 작업을 제출하지 않은 상태에서 `tokio::spawn`에 전달합니다.

```rust
    let num_batches = 4;
    let mut handles = vec![];

    for batch_idx in 0..num_batches {
        let w0 = w0.clone();       // Arc 복제: 약 1 ns
        let w1 = w1.clone();
        let module = module.clone();

        let batch_data: Vec<f32> = (0..DIM * DIM)
            .map(|i| ((i + batch_idx * 37) % 13) as f32 * 0.1)
            .collect();

        let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
            .and_then(move |(input, hidden, output)| {
                // 1단계: GEMM — hidden = input × W0
                // ... 실행을 구성하고 확정한 뒤 and_then으로 연결 ...
            })
            .and_then(move |(hidden, output, w1, module)| {
                // 2단계: MatVec — output = hidden × W1
                // ...
            })
            .and_then(move |(output, module)| {
                // 3단계: ReLU — result = max(0, output)
                // ...
            })
            .and_then(d2h);  // 4단계: 결과를 호스트로 복사

        handles.push(tokio::spawn(pipeline.into_future()));
    }
```

코드의 실행 흐름은 다음과 같습니다.

1. **`pipeline`은 `DeviceOperation`입니다.** 순방향 패스 전체를 표현하지만 GPU 작업은 하지 않습니다. 클로저와 구조체를 만드는 호스트 작업만 수행합니다.
2. **`.into_future()`는 이를 `DeviceFuture`로 변환합니다.** 이 시점에 스케줄링 정책이 스트림을 선택합니다. 라운드 로빈에 따라 배치 0은 스트림 0, 배치 1은 스트림 1을 받습니다.
3. **`tokio::spawn`은 퓨처를 Tokio 런타임에 맡깁니다.** 런타임은 실행 여유가 생기면 퓨처를 폴링합니다. 첫 폴링에서 파이프라인의 `execute()`가 호출되어 지정된 스트림에 GPU 작업을 제출합니다.
4. **호스트 런타임은 다른 일을 할 수 있습니다.** 첫 폴링 뒤 작업은 대기 상태가 되고, GPU는 네 스트림에서 연산합니다. 호스트 스레드가 결과를 기다리며 점유되지 않습니다.
5. **GPU 작업이 끝나면** `cuLaunchHostFunc` 콜백이 해당 Tokio 작업을 깨웁니다. 런타임이 다시 폴링해 `Vec<f32>` 결과를 전달합니다.

## 4단계: 결과 수집

```rust
    for (i, handle) in handles.into_iter().enumerate() {
        let result: Vec<f32> = handle.await??;
        println!("Batch {}: {} elements, first 4 = {:?}",
            i, result.len(), &result[..4]);
    }
```

`??`는 두 단계의 오류를 전파합니다. 바깥쪽 `JoinError`는 Tokio 작업이 패닉했을 때 발생하고, 안쪽 `DeviceError`는 GPU 작업이 실패했을 때 발생합니다. 실제 서비스 코드에서는 두 오류를 구분해 처리하는 편이 좋습니다.

## `.await`, `.sync()`, `tokio::spawn` 비교

각 방식은 다음 상황에 사용합니다.

**`.sync()`**는 GPU 작업이 끝날 때까지 호출 스레드를 블로킹합니다. 스크립트, 테스트, 비동기 런타임이 없는 곳에 적합합니다. 단순하지만 기다리는 동안 호스트 스레드는 다른 일을 하지 못합니다.

```rust
let result = pipeline.sync()?;
```

**`.await`**는 GPU가 작업하는 동안 현재 비동기 작업을 양보합니다. 같은 Tokio 스레드의 다른 작업이 진행될 수 있습니다. `.sync()`보다 처리량에 유리하지만 하나의 작업 안에서는 순차적입니다. `.await` 뒤 코드는 연산이 끝나야 실행됩니다.

```rust
let result = pipeline.await?;
```

**`tokio::spawn(op.into_future())`**는 파이프라인을 독립 작업으로 실행합니다. 생성한 쪽은 즉시 다음 작업으로 넘어가며, 나중에 조인 핸들로 결과를 받습니다. 여러 파이프라인을 서로 다른 스트림에서 동시에 실행할 때 사용하는 방식입니다.

```rust
let handle = tokio::spawn(pipeline.into_future());
// ... 다른 파이프라인을 생성하거나 다른 작업을 수행 ...
let result = handle.await??;
```

:::{tip}
순방향 패스처럼 서로 의존하는 연산을 연결할 때는 `.await`를 여러 번 호출하는 대신 `and_then` 체인을 사용하세요. 하나의 스트림에서 체인 전체가 실행되어 단계 사이의 스케줄링 오버헤드가 없습니다. `.await`를 반복하면 각 연산이 스케줄링 정책을 다시 거쳐 서로 다른 스트림에 배정될 수 있고, 이 경우 스트림 간 동기화가 필요할 수 있습니다.
:::

## 소유권 패턴

Rust의 소유권 규칙은 동시 GPU 프로그래밍을 안전하게 만들지만, 코드를 작성할 때 고려해야 할 점도 있습니다. 자주 사용하는 패턴을 살펴보겠습니다.

### `and_then` 클로저 사이에서 데이터 전달

각 `and_then` 클로저는 `move`로 값을 캡처합니다. 커널 실행이 `()`를 반환하므로 다음 단계에 필요한 버퍼는 명시적으로 전달해야 합니다.

```rust
launch_gemm(input, hidden, w0)
    .and_then(move |()| {
        // input은 커널이 소비합니다. hidden과 module은 캡처했지만
        // 소비하지 않았으므로 계속 사용할 수 있습니다.
        value((hidden, output, w1, module))
    })
```

클로저는 다음 단계에 필요한 모든 값을 튜플에 담은 `Value`를 반환합니다. 이 튜플이 단계 사이를 전달되는 바통 역할을 합니다.

### `Arc`로 불변 데이터 공유

모델 가중치, 조회 테이블처럼 여러 파이프라인에서 공유하는 불변 데이터는 `Arc`로 감싸는 것이 좋습니다. `.arc()` 조합기는 `DeviceOperation`의 출력을 자동으로 감쌉니다. 이미 가진 데이터는 `Arc::new()`로 감쌀 수 있습니다.

```rust
let weights = h2d(weight_data).arc().await?;  // Arc<DeviceBox<[f32]>>
for batch in batches {
    let w = weights.clone();  // 저렴한 참조 카운트 증가
    tokio::spawn(forward_pass(batch, w).into_future());
}
```

### 장치 메모리 수명 유지

GPU가 사용하는 동안 `DeviceBox`가 살아 있어야 합니다. `and_then` 체인에서는 클로저가 `DeviceBox`의 소유권을 가지므로 다음 단계가 넘겨받을 때까지 유지됩니다. 주의할 곳은 비동기 복사를 시작한 뒤 복사가 끝나기 전에 `DeviceBox`가 해제될 수 있는 `with_context` 클로저입니다.

```rust
fn d2h(dev: DeviceBox<[f32]>) -> impl DeviceOperation<Output = Vec<f32>> {
    with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let mut host = vec![0.0f32; dev.len()];
        unsafe {
            memcpy_dtoh_async(
                host.as_mut_ptr(), dev.cu_deviceptr(),
                dev.len() * std::mem::size_of::<f32>(),
                stream.cu_stream(),
            ).unwrap();
        }
        // dev는 클로저가 캡처하므로 클로저가 반환될 때까지 살아 있습니다.
        // 결과를 전달하기 전에 스트림이 동기화되어 비동기 복사가 끝납니다.
        value(host)
    })
}
```

핵심은 `dev`가 `move` 클로저에 캡처되어 클로저가 반환될 때까지 해제되지 않는다는 점입니다. `DeviceFuture`가 결과를 전달하기 전에 스트림을 동기화하므로 복사가 끝난 뒤에야 `dev`가 해제됩니다.

(concurrent-error-handling)=

## 오류 처리

GPU 작업의 오류도 다른 Rust 코드와 마찬가지로 `Result` 체인을 따라 전파됩니다. 여러 배치를 동시에 처리할 때는 하나가 실패해도 나머지를 계속 처리할 수 있습니다.

```rust
for (i, handle) in handles.into_iter().enumerate() {
    match handle.await {
        Ok(Ok(result)) => {
            println!("Batch {i}: {} elements", result.len());
        }
        Ok(Err(device_err)) => {
            eprintln!("Batch {i}: GPU error: {device_err}");
        }
        Err(join_err) => {
            eprintln!("Batch {i}: task panicked: {join_err}");
        }
    }
}
```

세 분기는 각각 성공, CUDA 드라이버 또는 스케줄링 오류(예: 메모리 부족, 잘못된 실행 설정), Tokio 작업 패닉을 처리합니다.

## 다중 GPU 실행

여러 GPU가 있는 시스템에서는 `init_device_contexts`에 더 큰 장치 수를 전달합니다.

```rust
init_device_contexts(0, 2)?;  // 기본 장치는 0, 장치 두 개를 사용할 수 있도록 준비
```

기본 장치를 GPU 0으로 설정하고 두 장치를 위한 스레드 로컬 맵을 준비합니다. 각 장치의 CUDA 컨텍스트, 스케줄링 정책, 스트림 풀은 처음 사용될 때 지연 생성됩니다. 정책을 사용하는 모든 작업(`.sync()`, `.await`)은 다른 장치를 명시하지 않는 한 기본 장치를 사용합니다. `ExecutionContext`는 장치 ID를 보관하므로 GPU 0의 작업은 GPU 1의 스트림에 영향을 주지 않습니다.

:::{tip}
다중 GPU 프로그래밍에서는 메모리 위치를 고려해야 합니다. GPU 0에서 할당한 `DeviceBox`는 피어 액세스를 활성화하지 않으면 GPU 1에서 사용할 수 없습니다. 장치별 작업을 구성할 때 `with_context`에서 `ctx.get_device_id()`를 확인하세요.
:::

## 성능 조정

### 스트림 풀 크기

기본 스트림 네 개는 대부분의 작업에 적합합니다. 처리량을 조금 더 끌어올리려면 작업 특성에 맞춰 풀 크기를 조정하세요.

| 작업 유형 | 권장 풀 크기 | 이유 |
|:--|:--|:--|
| 실행마다 큰 커널 하나 | 1–2 | 커널 하나가 이미 GPU를 포화시킵니다. |
| 작은 커널 다수 | 4–8 | 스트림 간 실행 오버헤드를 겹칠 수 있습니다. |
| 커널과 메모리 복사 혼합 | 2–4 | 연산과 데이터 전송을 겹칩니다. |
| 지연 시간에 민감한 서빙 | 요청당 1 | 선두 요청 때문에 뒤 작업이 막히는 현상을 줄입니다. |

### 프로파일링

Nsight Systems는 스트림 점유 상태를 시각화할 때 사용하는 표준 도구입니다.

```bash
nsys profile --trace=cuda cargo oxide run async_mlp
nsys-ui report.nsys-rep
```

같은 스트림의 커널 사이 빈 구간(실행 오버헤드), 풀 안에서 유휴 상태인 스트림(작업 분배 불균형), 스트림 사이의 예상치 못한 직렬화(누락되거나 불필요한 의존성)를 살펴보세요.

### 자주 발생하는 문제

| 문제 | 증상 | 해결 방법 |
|:--|:--|:--|
| 비동기 컨텍스트에서 `.sync()` 사용 | Tokio 스레드가 막혀 그 위의 모든 작업이 지연됩니다. | `.await`를 사용합니다. |
| 스트림 동기화 전에 `DeviceBox` 해제 | 해제된 메모리 사용으로 인한 충돌 또는 손상된 결과 | `and_then` 클로저가 소유권을 유지하도록 합니다. |
| 스트림이 너무 많음 | 스케줄링 비용이 중첩 실행의 이득보다 커집니다. | 프로파일링한 뒤 풀 크기를 줄입니다. |
| `init_device_contexts` 누락 | 첫 작업에서 `DeviceError::Context`가 발생합니다. | 프로그램 시작 시 한 번 호출합니다. |

:::{seealso}
전체 `async_mlp` 소스와 빌드 안내, 예상 출력은 [비동기 MLP 파이프라인](../projects/async-mlp-pipeline) 장을 참고하세요.
:::

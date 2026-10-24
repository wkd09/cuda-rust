# 스케줄링과 스트림

앞선 장에서는 GPU 작업을 설명하는 어휘인 `DeviceOperation`, `and_then` 연결, `zip!` 묶음을 살펴보았습니다. 하지만 작업 설명이 실제 실행은 아닙니다. 어느 순간에는 이 조리법을 주방에 전달해야 합니다. 이 장에서는 CUDA 스트림과 cuda-oxide 스케줄링 정책이 스트림에 작업을 배정하는 방식, `DeviceOperation`이 실행 중인 `DeviceFuture`가 될 때 내부에서 일어나는 일을 설명합니다.

:::{seealso}
스케줄링 정책의 기반이 되는 CUDA 스트림 의미는 [CUDA 프로그래밍 가이드 — 스트림](https://docs.nvidia.com/cuda/cuda-programming-guide/#streams)을 참조하세요.
:::

## 계산대 차선

CUDA 스트림을 식료품점의 **계산대 차선**이라고 생각해 보세요. 각 차선은 줄 선 순서대로 고객을 처리합니다. 먼저 도착한 고객이 먼저 계산합니다. 여러 차선은 서로 독립적으로 운영되므로 한 차선에서 계산하는 동안 다른 차선은 별도 주문을 포장할 수 있습니다. 서로 다른 차선의 작업이 시간상 겹치면 매장의 처리량이 늘어납니다.

GPU도 같은 방식입니다. **스트림**은 순서가 보장되는 작업 큐입니다. 하나의 스트림에서는 모든 작업이 순차 실행됩니다. 커널 A가 끝난 다음 커널 B가 시작합니다. 그러나 커널 A를 스트림 0에, 커널 B를 스트림 1에 넣으면 하드웨어에서 겹쳐 실행될 수 있습니다.

```text
스트림 0:  ┌─── 커널 A ───┐
           └──────────────┘
스트림 1:       ┌─── 커널 B ───┐
                └──────────────┘

           ◄─── 시간 ────────────────►
```

스트림이 하나면 모든 작업이 직렬로 실행됩니다. 여러 스트림에서는 독립 작업이 겹쳐 실행되어 GPU가 더 바쁘게 작동합니다. 이것이 GPU 동시성의 기본 메커니즘입니다.

## 스트림을 직접 다룰 일이 드문 이유

CUDA C++에서는 프로그래머가 스트림을 만들고 각 작업에 스트림을 지정하며 다른 스트림의 결과에 의존하는 작업 사이에 이벤트를 직접 넣습니다. 강력하지만 번거롭고 각 함수를 특정 동시성 전략에 묶습니다.

cuda-oxide는 **스케줄링 정책**이라는 간접 계층을 둡니다. 직접 스트림을 고르는 대신 `DeviceOperation`을 정책에 넘기면 정책이 스트림을 선택합니다. 따라서 파이프라인 코드 자체를 바꾸지 않고도 같은 파이프라인을 디버깅용 단일 스트림, 처리량 향상을 위한 스트림 풀, 직접 작성한 사용자 정의 정책으로 실행할 수 있습니다.

## `SchedulingPolicy` 트레이트

스케줄링 정책은 “이 작업을 어느 스트림에서 실행할까?”라는 질문에 답합니다. 트레이트는 세 메서드로 구성됩니다.

- **`init`** — 시작 시 한 번 호출해 CUDA 스트림을 만듭니다.
- **`schedule`** — 스트림을 선택하고 `.await`용 `DeviceFuture`로 작업을 감쌉니다.
- **`sync`** — 스트림을 선택해 작업을 실행하고 완료될 때까지 블로킹합니다.

정책은 `Sync`입니다. 장치의 모든 작업이 인스턴스 하나를 공유하므로 스트림 선택은 스레드 안전해야 합니다.

## 기본 정책 `StreamPoolRoundRobin`

`init_device_contexts(0, 1)`을 호출하면 cuda-oxide는 CUDA 스트림 네 개를 가진 `StreamPoolRoundRobin`을 만듭니다. 작업을 스케줄링할 때마다 원자 카운터를 증가시키고 풀의 다음 스트림을 선택합니다.

```text
작업 1  →  스트림 0  ──► ████████
작업 2  →  스트림 1  ──►    ████████         (1과 중첩)
작업 3  →  스트림 2  ──►       ████████      (1, 2와 중첩)
작업 4  →  스트림 3  ──►          ████████
작업 5  →  스트림 0  ──►                ████████  (1을 기다림)
```

선택은 잠금이 필요 없습니다. 풀 크기로 나눈 나머지를 사용하는 `AtomicUsize`의 `fetch_add` 한 번으로 처리합니다. GPU 작업 비용과 비교하면 오버헤드는 무시할 수 있습니다.

스트림 네 개는 좋은 기본값입니다. 과도한 컨텍스트 전환 오버헤드 없이 커널 실행과 메모리 전송을 겹칠 만큼 GPU에 충분한 진행 중 작업을 제공합니다. 대부분의 워크로드에서는 스트림 수를 직접 고려할 필요가 없습니다. 정책이 알아서 처리합니다.

### 라운드 로빈이 효과적인 경우

- **배치 추론:** 배치마다 독립 파이프라인이 있습니다. 라운드 로빈이 배치를 스트림별로 분배해 계산을 겹칩니다.
- **계산과 전송 혼합:** 한 스트림이 커널을 실행하는 동안 다른 스트림은 데이터를 복사합니다. GPU 복사 엔진과 연산 장치가 동시에 작동합니다.
- **작은 커널 다수:** 실행을 겹쳐 시작 오버헤드 때문에 커널 사이에 생기는 간격을 줄이고 GPU를 더 바쁘게 유지합니다.

### 다른 정책을 고려할 때

- **의존성이 많은 연결:** 앞선 장의 순방향 패스처럼 `and_then` 연결 하나로 구성하면 전체 연결은 어차피 스트림 하나에서 실행됩니다. 라운드 로빈은 *서로 독립적인* 작업 여러 개를 스케줄링할 때만 의미가 있습니다.
- **매우 큰 커널:** GPU를 이미 포화시키는 커널 하나는 다중 스트림 스케줄링으로 이득을 얻지 않습니다. 추가 스트림은 유휴 상태로 남습니다.

## `SingleStream` — 단일 차선, 엄격한 순서

:::{note}
`SingleStream`은 스케줄링 내부에 구현되어 있지만 현재 `GlobalSchedulingPolicy`에 연결되지 않았고 `init_device_contexts`를 통해 노출되지 않습니다. 기본 설정은 항상 `StreamPoolRoundRobin`을 사용합니다. 이 절은 향후 API의 설계 의도를 설명합니다.
:::

디버깅하거나 **모든** 작업 사이의 순서를 보장해야 할 때 `SingleStream`은 작업을 스트림 하나로 보냅니다. 모든 작업은 앞선 작업의 결과를 확인하므로 스트림 동시성에서 비롯된 버그 가능성이 사라집니다.

```text
작업 1  →  스트림 0  ──► ████████
작업 2  →  스트림 0  ──►          ████████   (1을 기다림)
작업 3  →  스트림 0  ──►                  ████████
```

:::{tip}
GPU 파이프라인의 동시성 버그가 의심되면 `SingleStream`으로 바꾸어 확인하는 것이 가장 빠릅니다. 버그가 사라지면 서로 다른 스트림의 작업 사이에 의존성이 빠진 것입니다. 그대로라면 원인은 다른 곳에 있습니다.
:::

## 런타임 설정

비동기 작업을 실행하기 전에 스레드 로컬 장치 컨텍스트를 초기화합니다.

```rust
use cuda_async::simt::device_context::init_device_contexts;

init_device_contexts(0, 1)?;
```

첫 번째 인자는 기본 GPU 번호이고 두 번째는 관리할 장치 수입니다. 내부적으로 스레드 로컬을 등록하며, 장치를 처음 사용할 때 `StreamPoolRoundRobin`을 지연 생성합니다. 기본 풀에는 스트림 네 개가 있습니다.

프로그램 시작 시 `.sync()` 또는 `.await`를 호출하기 전에 한 번 초기화하세요. 같은 스레드에서 두 번 호출하면 오류가 반환됩니다.

## `.await`에서 일어나는 일

```{figure} images/device-future-polling.svg
:align: center
:width: 100%

왼쪽: `DeviceFuture`의 3상태 기계입니다. 첫 poll에서 GPU 작업과 `cuLaunchHostFunc` 콜백을 제출한 뒤 `Poll::Pending`을 반환합니다. GPU가 완료되면 콜백이 `AtomicBool`을 설정하고 작업을 깨웁니다. 두 번째 poll에서 결과를 전달합니다. 오른쪽: `.sync()`(전체 시간 동안 스레드 차단)와 `.await`(GPU 작업 중 스레드가 다른 작업 실행)의 비교입니다.
```

구성에서 완료까지 작업 하나가 거치는 전체 과정은 다음과 같습니다.

```text
module.kernel_async(&prepared, ...)  ← 검사된 작업 계획 생성 (GPU 작업 없음)
        │
        ▼
  AsyncKernelLaunch            ← 지연 실행되고 스트림과 무관한 DeviceOperation
        │
        │  .await
        ▼
  IntoFuture::into_future()    ← 스케줄링 정책이 스트림 선택
        │
        ▼
  DeviceFuture                 ← 스트림에 바인딩되어 poll 준비 완료
        │
        │  첫 poll()
        ▼
  stream 2에서 execute()      ← GPU 작업 제출
  stream 2에서 cuLaunchHostFunc ← 커널 뒤에 호스트 콜백 큐잉
  Poll::Pending 반환
        │
        │  ... GPU 실행 중, 호스트 스레드는 비어 있음 ...
        │
        │  CUDA 드라이버 스레드에서 콜백 실행
        │  → AtomicBool 설정, AtomicWaker 깨우기
        │
        │  두 번째 poll()
        ▼
  Poll::Ready(Ok(())) 반환    ← 호출자에게 결과 전달
```

핵심은 첫 `poll()`과 콜백 사이에 **점유된 호스트 스레드가 없다는 것**입니다. 비동기 런타임은 작업을 대기 상태로 두고 다른 작업을 실행합니다. GPU는 완료되면 `cuLaunchHostFunc` 콜백을 통해 런타임에 알립니다. 이 때문에 동시 워크로드에서 `.await`는 `.sync()`보다 확장성이 좋습니다. 작업마다 스레드를 하나씩 점유하지 않고도 수십 개의 작업을 동시에 진행할 수 있습니다.

## 스트림 수동 제어

대부분은 스케줄링 정책이 스트림을 관리합니다. 하지만 특정 스트림을 요구하는 CUDA 라이브러리와 상호 운용하거나, 계산과 전송을 세밀하게 겹치거나, 커널 하나만 프로파일링할 때는 직접 제어해야 합니다. 이 경우를 위해 cuda-oxide는 전체 스트림 API를 제공합니다.

### 스트림 만들기

```rust
let ctx = CudaContext::new(0)?;
let default = ctx.default_stream();  // the per-context default (null) stream
let custom  = ctx.new_stream()?;     // a new non-blocking stream
```

기본 스트림은 CUDA에서 특별한 동기화 의미를 가집니다. 대부분의 다른 스트림과 암묵적으로 직렬화됩니다. `new_stream()`이 만드는 비동기(non-blocking) 스트림에는 이 제약이 없으므로 스케줄링 정책은 해당 스트림만 사용합니다.

### 분기와 합류

자식 스트림을 부모 스트림에서 분기해 독립 작업을 실행하고 결과를 다시 합류하는 패턴이 흔합니다. `fork`는 부모의 현재 지점에 암묵적 의존성이 있는 스트림을 새로 만듭니다. 따라서 부모의 이전 작업이 모두 끝난 다음 자식이 시작됩니다. `join`은 반대로 부모가 진행하기 전에 자식 완료를 기다립니다.

```rust
let main = ctx.default_stream();

// Upload data on main
let buf_a = DeviceBuffer::from_host(&main, &data_a)?;
let buf_b = DeviceBuffer::from_host(&main, &data_b)?;

// Fork: children see the uploads
let child_1 = main.fork()?;
let child_2 = main.fork()?;

// SAFETY: cfg matches process's indexing/resources and both buffers' bounds.
// Run independent work in parallel.
unsafe {
    module.process(&child_1, cfg, &mut buf_a)?;
    module.process(&child_2, cfg, &mut buf_b)?;
}

// Join: main waits for both children
main.join(&child_1)?;
main.join(&child_2)?;

// SAFETY: cfg matches combine's indexing/resources and both input bounds.
unsafe { module.combine(&main, cfg, &buf_a, &buf_b) }?;
```

GPU의 시간 흐름은 다음과 같습니다.

```text
main:      ██ upload_a ██ upload_b ██ ──fork──────────────── join ──► ██ combine ██
                                         |                    ^  ^
child_1:                                 └─► ██ process_a ██ ─┘  |
                                         |                       |
child_2:                                 └─► ██ process_b ██ ────┘
```

내부에서 `fork`와 `join`은 CUDA 이벤트를 사용합니다. 한 스트림에서 `cuEventRecord`를 호출하고 다른 스트림에서 `cuStreamWaitEvent`를 호출합니다. 이벤트는 GPU 측 동기화 토큰이므로 분기나 합류 동안 호스트 스레드를 블로킹하지 않습니다.

### 세밀한 순서를 위한 이벤트

`fork` / `join`이 너무 거친 단위라면 이벤트를 직접 사용해 서로 다른 스트림의 특정 지점 사이에 순서를 설정할 수 있습니다.

```rust
// Record an event on stream A after the kernel finishes
let event = stream_a.record_event(None)?;

// Stream B waits for that specific point before proceeding
stream_b.wait(&event)?;
```

CUDA 이벤트는 스트림 소유가 아니라 독립적인 동기화 토큰입니다. `record`는 스트림 하나의 특정 지점에 이벤트를 기록하고, `wait`는 다른 스트림에 의존성을 삽입합니다. 이벤트는 만남 지점입니다. 스트림 A에서 이벤트가 발생하기 전까지 스트림 B에서 `wait` 뒤에 있는 작업은 실행되지 않습니다.

이벤트는 GPU 실행 시간을 측정하는 표준 방법이기도 합니다. 시간 측정에는 `CU_EVENT_DISABLE_TIMING` 없이 생성된 이벤트가 필요합니다. 명시적인 플래그를 전달해 타이밍을 켜세요.

```rust
// bindgen flattens the CUDA enums, so the constant is a crate-root item
// named after its enum, not a variant inside a module.
use cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT as CU_EVENT_DEFAULT;

let start = stream.record_event(Some(CU_EVENT_DEFAULT))?;
// SAFETY: config and arguments satisfy my_kernel's requirements.
unsafe { module.my_kernel(&stream, config) }?;
let end = stream.record_event(Some(CU_EVENT_DEFAULT))?;
end.synchronize()?;
println!("Kernel took {:.2} ms", start.elapsed_ms(&end)?);
```

이 측정값은 호스트 측 스케줄링 오버헤드가 아니라 실제 GPU 시간을 나타냅니다. `record_event(None)`은 기본적으로 타이밍을 끈 이벤트를 만들며 `elapsed_ms`에 사용할 수 없습니다.

:::{tip}
일반적인 비동기 파이프라인에서는 스트림, 이벤트, 분기/합류를 직접 만들 필요가 없습니다. 스케줄링 정책과 `and_then` 연결이 순서를 자동으로 관리합니다. 수동 API는 상호 운용, 프로파일링, 고급 최적화에 사용합니다.
:::

## 적합한 방식 선택

| 상황 | 권장 방식 |
|:---|:---|
| 단순 스크립트, 계약이 지정된 커널 하나 | `module.kernel_async(&prepared, ...).sync()` |
| 여러 단계 파이프라인 (GEMM → ReLU → D2H) | `and_then` 연결, 정책이 스트림 하나를 선택 |
| 독립 배치를 동시에 실행 | 배치마다 `tokio::spawn`, 라운드 로빈 분배 |
| 스트림 순서 버그 의심 시 디버깅 | `SingleStream`으로 전환 |
| 기존 CUDA 라이브러리와 상호 운용 | 해당 라이브러리 스트림에 `.sync_on(&stream)` |
| 커널 하나만 프로파일링 | 실행 전후에 이벤트 명시 |
| 최대 처리량 | Nsight Systems로 프로파일링하고 풀 크기 조정 |

:::{seealso}
CUDA 이벤트의 전체 사양은 [CUDA 프로그래밍 가이드 — 이벤트](https://docs.nvidia.com/cuda/cuda-programming-guide/#events)를 참조하세요. 이 스케줄링 개념을 여러 배치 워크로드에 적용하는 방법은 [동시 실행](concurrent-execution) 장을 참조하세요.
:::

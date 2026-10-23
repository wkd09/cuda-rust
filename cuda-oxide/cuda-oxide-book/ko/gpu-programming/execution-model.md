# CUDA 실행 모델

이 장에서는 CUDA SIMT 실행 모델, 즉 스레드, 워프, 블록, 그리드로 작업을 구성하는 방법과 cuda-oxide가 각 수준을 안전하고 편리한 Rust API로 제공하는 방식을 설명합니다.

:::{seealso}
CUDA 실행 모델의 권위 있는 설명은 [CUDA 프로그래밍 가이드 — 프로그래밍 모델](https://docs.nvidia.com/cuda/cuda-programming-guide/#programming-model)을 참조하세요.
:::

## 스레드, 블록, 그리드

모든 커널 실행은 **그리드**를 만들고 그 안에 **스레드 블록**을 둡니다. 다음 3단계 계층은 GPU 프로그래밍의 기초입니다.

| 수준 | 설명 | 크기 | 주요 특성 |
|:---|:---|:---|:---|
| **그리드** | 한 번의 커널 호출로 실행되는 모든 블록 | 차원별 최대 2³¹ - 1개 블록 | 블록은 서로 독립적으로 실행 |
| **블록** | 협력할 수 있는 스레드 그룹 | 최대 1024개 스레드 | 스레드가 빠른 온칩 메모리를 공유 |
| **워프** | 블록 안에서 연속된 32개 스레드 | 항상 32개 | 같은 명령을 동시에 실행(SIMT) |

커널 실행에는 그리드의 블록 수(**그리드 차원**)와 블록마다 실행할 스레드 수(**블록 차원**)를 지정합니다. 하드웨어는 연속된 스레드 32개를 자동으로 묶어 워프를 구성합니다. 워프를 직접 만들 필요는 없습니다.

```{figure} images/simt-thread-hierarchy.svg
:align: center
:width: 100%

CUDA 스레드의 3단계 계층입니다. 2×2 그리드의 각 블록은 256개 스레드로 구성되며, 32개씩 8개 워프를 이룹니다. 아래 범례는 CUDA 개념과 cuda-oxide API의 대응 관계를 보여줍니다.
```

## cuda-oxide의 스레드 인덱싱

커널 안에서 각 스레드는 자신이 어느 원소를 처리할지 알아야 합니다. CUDA는 내장 변수(`threadIdx`, `blockIdx`, `blockDim`, `gridDim`)를 제공하며 cuda-oxide는 이를 `cuda_device::thread` 모듈로 감쌉니다.

```rust
use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

`thread::index_1d()`는 `blockIdx.x * blockDim.x + threadIdx.x`를 계산합니다. 그리드와 블록의 Y/Z 차원이 1이면 스레드를 서로 다른 배열 원소에 대응시킵니다. 이는 1차원 데이터 병렬 커널에서 흔히 사용하는 형태이며 `domain = 1`인 준비된 실행이 이를 증명합니다.

개별 구성요소가 필요한 경우 cuda-oxide는 다음 원시 접근자도 제공합니다.

| cuda-oxide API | CUDA C++ 대응식 | 반환값 |
|:---|:---|:---|
| `thread::index_1d()` | `blockIdx.x * blockDim.x + threadIdx.x` | 전역 1차원 스레드 인덱스 |
| `thread::threadIdx_x()` | `threadIdx.x` | 블록 내부 스레드 위치 |
| `thread::blockIdx_x()` | `blockIdx.x` | 그리드 내부 블록 위치 |
| `thread::blockDim_x()` | `blockDim.x` | 블록의 x축 스레드 수 |

:::{tip}
2차원 행렬 연산처럼 다차원 인덱싱이 필요하면 `_x` 변형과 함께 `threadIdx_y()`, `blockIdx_y()`, `blockDim_y()`를 사용해 행과 열 인덱스를 계산하세요.
:::

## 워프와 SIMT 실행

**워프**는 NVIDIA GPU의 기본 스케줄링 단위입니다. 블록 안의 연속된 스레드 32개가 워프 하나를 이루고, 워프의 32개 스레드는 **같은 명령**을 **같은 시간에** 실행하지만 **서로 다른 데이터**를 처리합니다. 이 모델을 **SIMT**(Single Instruction, Multiple Thread)라고 합니다.

워프의 모든 스레드가 같은 제어 흐름을 따르면 전체 처리량을 낼 수 있습니다. 스레드마다 서로 다른 `if` 분기를 선택해 실행 경로가 갈라지면 하드웨어는 한 분기에서 다른 스레드를 마스킹한 채 실행한 다음, 다른 분기를 실행하고 다시 합류합니다. 이를 **분기 발산**이라고 하며 처리량을 직접 낮춥니다.

```{figure} images/simt-warp-execution.svg
:align: center
:width: 100%

왼쪽: 32개 스레드 모두가 한 사이클에 같은 명령을 실행하는 균일 실행. 오른쪽: 짝수·홀수 스레드가 다른 경로를 선택해 두 번의 순차 실행이 필요한 분기 발산.
```

### 왜 알아야 할까요?

올바른 커널을 작성할 때 워프의 세부 사항을 모두 고려할 필요는 없습니다. cuda-oxide가 처리합니다. 하지만 SIMT를 이해하면 더 빠른 커널을 작성할 수 있습니다.

- **제어 흐름을 균일하게 유지하세요.** 워프의 모든 스레드가 같은 분기를 선택하면 발산에 따른 손실이 없습니다.
- **데이터 의존 분기도 괜찮습니다.** 서로 가까운 스레드, 즉 같은 워프의 스레드가 같은 경로를 선택하는 경향이 있다면 됩니다.
- **핫 루프에서 스레드 ID에 따른 분기를 피하세요.** `if thread::threadIdx_x() % 2 == 0` 같은 코드는 모든 워프의 분기를 확실히 발산시킵니다.

:::{seealso}
워프 실행과 재합류의 전체 하드웨어 사양은 [CUDA 프로그래밍 가이드 — SIMT 아키텍처](https://docs.nvidia.com/cuda/cuda-programming-guide/#simt-architecture)를 참조하세요.
:::

## 하드웨어 매핑

커널을 실행하면 GPU 하드웨어 스케줄러가 각 블록을 **스트리밍 멀티프로세서**(SM)에 배정합니다. 하나의 SM에서 여러 블록이 동시에 실행될 수 있으며, 동시 실행 수는 레지스터, 공유 메모리, 스레드 등 블록의 리소스 사용량에 따라 달라집니다.

핵심은 **그리드와 블록 차원은 프로그래머가 정하지만 나머지는 하드웨어가 제어한다**는 점입니다. 어떤 SM이 어떤 블록을 실행할지, 블록 실행 순서가 무엇인지 지정하지 않습니다. 이 분리 덕분에 같은 커널을 SM이 몇 개뿐인 노트북 GPU부터 100개가 넘는 데이터센터 GPU까지 확장할 수 있습니다.

```{figure} images/simt-hardware-mapping.svg
:align: center
:width: 100%

GPU 스케줄러가 네 개 SM에 블록 여덟 개를 배정한 모습입니다. 각 SM은 자체 워프 스케줄러, CUDA 코어, 공유 메모리/L1 캐시를 가집니다. 점선 화살표의 블록 4–7은 블록 0–3이 끝난 뒤 실행되거나 리소스가 허용되면 동시에 대기열에 배정됩니다.
```

### 동시 실행을 제한하는 요소

각 SM은 고정된 리소스 풀을 가집니다. 다음 리소스가 **모두** 충분할 때만 블록을 SM에 배정할 수 있습니다.

| 리소스 | 일반적인 Ampere 한도 | 제어하는 설정 |
|:---|:---|:---|
| 스레드 | SM당 2048개 | `block_dim` |
| 레지스터 | SM당 65536개 | 컴파일러 할당 |
| 공유 메모리 | SM당 164 KB (구성 가능) | `shared_mem_bytes` |
| 블록 슬롯 | SM당 32개 | 그리드 크기 |

블록이 끝나면 리소스를 반환하고 스케줄러는 대기 중인 블록을 해당 SM에 바로 배정합니다. GPU의 SM 수보다 **더 많은** 블록을 실행하는 것은 단순히 허용되는 수준이 아니라 정상적이고 일반적인 방식입니다.

:::{seealso}
아키텍처별 SM 리소스 한도와 점유율 계산은 [CUDA 프로그래밍 가이드 — 하드웨어 구현](https://docs.nvidia.com/cuda/cuda-programming-guide/#hardware-implementation)을 참조하세요.
:::

## 실행 구성

호스트 측 `LaunchConfig`는 런타임에 그리드 모양을 알려줍니다.

```rust
use cuda_core::simt::LaunchConfig;

// 간단한 1차원 실행: 블록당 스레드 256개, N개 원소를 처리할 충분한 블록 수
let cfg = LaunchConfig::for_num_elems(N as u32);
```

`for_num_elems`는 블록 크기를 256으로 두고 올림 나눗셈으로 그리드 크기를 계산합니다. 대부분의 원소별 커널에 적합한 기본값입니다. 더 세밀하게 설정하려면 `LaunchConfig`를 직접 구성하세요.

```rust
let cfg = LaunchConfig {
    grid_dim: (4, 4, 1),      // 4×4 = 16개 블록
    block_dim: (16, 16, 1),   // 블록마다 16×16 = 256개 스레드
    shared_mem_bytes: 0,       // 동적 공유 메모리 사용 안 함
};
```

그런 다음 생성된 원시 실행 메서드에 전달합니다. `LaunchConfig` 자체는 숨겨진 Y/Z 차원이 비활성 상태인지 증명하지 않으므로 호출은 unsafe입니다.

```rust
// SAFETY: for_num_elems는 1차원 실행이며 세 버퍼의 길이는 모두 N입니다.
unsafe {
    module.vecadd(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )
}
.expect("Kernel launch failed");
```

비동기 API에서도 다음과 같이 실행할 수 있습니다.

```rust
// SAFETY: 1차원 실행이며 버퍼 길이는 N이고 모듈과 스케줄러는 같은 컨텍스트를 사용합니다.
unsafe {
    module.vecadd_async(
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )
}?
.sync()?;
```

(execution-choosing-block-size)=

### 블록 크기 선택

블록 크기는 가장 중요한 튜닝 변수입니다.

- **256개 스레드**는 안전한 기본값입니다. 대부분 아키텍처에서 여러 블록을 SM에 올리는 점유율과 레지스터 부담이 균형을 이룹니다.
- **2의 거듭제곱**(128, 256, 512)은 워프 경계에 자연스럽게 맞아 스레드를 낭비하지 않습니다.
- **너무 작으면**(128 미만) 워프 스케줄러를 충분히 활용하지 못할 수 있습니다.
- **너무 크면**(1024) 블록 스레드 한도를 모두 사용하므로 SM당 동시 블록 수가 줄어들 수 있습니다.

그리드 크기는 블록 크기와 문제 크기로 결정됩니다. `grid_x = (N + block_x - 1) / block_x`이며 `LaunchConfig::for_num_elems`가 바로 이 값을 계산합니다.

### 경계 검사

그리드 크기를 올림하므로 일부 스레드 인덱스가 배열 길이를 벗어납니다. cuda-oxide의 `DisjointSlice`는 이를 안전하게 처리합니다. 범위를 벗어난 인덱스에 대해 `get_mut`은 `None`을 반환하므로 해당 스레드는 아무 작업도 하지 않습니다.

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {   // 범위를 벗어난 스레드는 건너뜀
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

이는 경계 검사를 프로그래머가 책임지는 CUDA C++와 의도적으로 다릅니다. cuda-oxide는 분기 하나를 추가하는 대신(마지막 블록을 제외하면 워프 전체에서 균일한 분기) 범위를 벗어난 메모리 버그 유형 전체를 제거합니다.

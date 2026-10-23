# 워프 수준 프로그래밍

**워프**는 CUDA의 기본 스케줄링 단위로, 같은 SM에서 같은 명령을 동시에 실행하는 스레드 32개입니다. 32개 스레드가 명령 포인터를 공유하므로 **워프 셔플** 명령을 사용해 데이터를 직접 교환할 수 있습니다. 레지스터 간 전송은 대략 한 사이클이 걸리며 공유 메모리, 배리어, 동기화가 필요하지 않습니다.

cuda-oxide는 `cuda_device::warp`를 통해 워프 intrinsic 전체를 제공합니다. 이 장에서는 셔플과 투표 연산 및 이를 활용한 워프 축약, 브로드캐스트, 스캔, ballot 기반 필터링 패턴을 다룹니다.

:::{seealso}
PTX 인코딩과 전체 폭 변형 목록은 [CUDA 프로그래밍 가이드 — 워프 셔플 함수](https://docs.nvidia.com/cuda/cuda-programming-guide/#warp-shuffle-functions)를 참조하세요.
:::

---

## 레인과 워프 식별

블록의 모든 스레드는 워프 안에서 **레인 ID**(0–31)를 가집니다.

```rust
use cuda_device::warp;

let lane = warp::lane_id();     // 0..31, hardware register %laneid
let warp = warp::warp_id();     // threadIdx.x / 32
```

`warp_id()`는 `threadIdx.x / 32`에서 계산됩니다. 2차원 또는 3차원 블록에서는 이 값이 하드웨어 워프를 나타내지 않습니다. 워프는 블록 선형 ID가 연속된 스레드로 구성되며 ID는 `x + blockDim.x * (y + blockDim.y * z)`입니다. 이런 블록에서는 `this_thread_block().thread_rank() / 32`를 사용하세요.

---

## 셔플: 레지스터 간 데이터 교환

셔플 변형 네 가지를 사용하면 메모리를 거치지 않고 다른 스레드의 레지스터를 읽을 수 있습니다.

| 함수 | 동작 | 원본 레인 |
|:---|:---|:---|
| `shuffle(val, src)` | 지정한 레인에서 읽기 | `src` |
| `shuffle_xor(val, mask)` | `lane_id ^ mask` 레인에서 읽기 | `lane_id ^ mask` |
| `shuffle_down(val, delta)` | `lane_id + delta` 레인에서 읽기 | `lane_id + delta` |
| `shuffle_up(val, delta)` | `lane_id - delta` 레인에서 읽기 | `lane_id - delta` |

각 변형은 `u32`와 `f32` 모두 지원합니다.

```rust
let partner_val = warp::shuffle_xor_f32(my_val, 1);
let broadcast   = warp::shuffle_f32(my_val, 0);     // lane 0's value to all
let neighbor    = warp::shuffle_down_f32(my_val, 1); // next lane's value
```

모든 셔플은 **워프 동기식**이며 워프를 암묵적으로 동기화합니다. `sync_threads()`는 필요하지 않습니다. 셔플 패턴 안에서 호출하면 불필요하고 낭비가 됩니다.

---

## 워프 축약

가장 흔한 셔플 패턴은 **버터플라이 축약**입니다. 5단계(⌈log₂(32)⌉ = 5)만 거치면 모든 레인이 값 32개의 합(또는 최솟값, 최댓값 등)을 누적합니다. 공유 메모리도 배리어도 필요 없이 셔플 다섯 번으로 끝납니다.

```{figure} images/warp-shuffle-reduction.svg
:align: center
:width: 100%

`shuffle_xor`를 사용한 버터플라이 축약입니다. 각 단계에서 레인은 XOR 파트너와 값을 교환해 더합니다. 다섯 단계(마스크 16, 8, 4, 2, 1) 후 레인 0은 값 32개의 합을 보유합니다.
```

```rust
use cuda_device::warp;

fn warp_reduce_sum(mut val: f32) -> f32 {
    val += warp::shuffle_xor_f32(val, 16);
    val += warp::shuffle_xor_f32(val, 8);
    val += warp::shuffle_xor_f32(val, 4);
    val += warp::shuffle_xor_f32(val, 2);
    val += warp::shuffle_xor_f32(val, 1);
    val
}
```

축약이 끝나면 **레인 32개 모두** 합을 보유합니다(XOR은 대칭이므로 양쪽 파트너가 모두 값을 누적합니다). 결과가 레인 0에만 필요하다면 `shuffle_down`을 사용할 수 있습니다.

```rust
fn warp_reduce_sum_lane0(mut val: f32) -> f32 {
    val += warp::shuffle_down_f32(val, 16);
    val += warp::shuffle_down_f32(val, 8);
    val += warp::shuffle_down_f32(val, 4);
    val += warp::shuffle_down_f32(val, 2);
    val += warp::shuffle_down_f32(val, 1);
    val
}
```

`shuffle_down`에서는 레인 0만 올바른 결과를 보유하고 나머지 레인은 부분합을 가집니다. 레인 0만 출력에 쓴다면 문제가 없습니다.

:::{tip}
블록 전체 축약이 필요하신가요? 셔플로 워프 안에서 축약하고 워프별 결과 32개를 공유 메모리에 쓴 뒤 `sync_threads()`로 동기화하고 마지막 워프 하나에서 워프별 결과를 축약하세요. 이 혼합 방식은 배리어 5단계를 없애므로 공유 메모리만 사용하는 트리보다 빠릅니다.
:::

### Ampere 정수 연산: 명령 하나로

위의 모든 축약, 즉 직접 작성한 버터플라이와 제네릭 `cooperative_groups::warp_reduce`는 셔플 다섯 번과 결합 연산 다섯 번이 필요합니다. `sm_80` 이상에서 **정수** 축약은 하드웨어 명령 하나인 `redux.sync`로 수행할 수 있습니다. 연산별 함수는 다음과 같습니다.

| 함수 | 축약 연산 |
|:---|:---|
| `warp::redux_sync_add(mask, v)` | wrapping 합, `u32` |
| `warp::redux_sync_min_u32` / `_min_i32` | 최솟값 |
| `warp::redux_sync_max_u32` / `_max_i32` | 최댓값 |
| `warp::redux_sync_and` / `_or` / `_xor` | 비트 연산 |

```rust
use cuda_device::warp;

const FULL_MASK: u32 = 0xffff_ffff;

// The whole reduction. One instruction, every lane holds the result.
let total = warp::redux_sync_add(FULL_MASK, my_value);
```

A10G(`sm_86`)에서 스레드 100만 개가 각각 전체 워프 `u32` 합 축약 64회를 수행하도록 측정했습니다. 버터플라이 방식은 실행 한 번당 159.4 µs, `redux.sync` 방식은 33.6 µs로 결과는 비트 단위까지 같으면서 **4.74배** 빨랐습니다. `u32` 덧셈은 2³² 모듈러 연산으로 결합법칙이 성립하고 최솟값/최댓값/and/or/xor도 정확한 연산이므로 축약 순서가 결과를 바꾸지 않습니다.

이 수치는 연산 원시 요소만 따로 측정한 것입니다. 측정 커널은 대부분 축약만 수행합니다. 메모리 바운드 패스가 끝날 때 한 번 축약하는 커널은 개선 폭이 작습니다. 내부 루프에서 축약하는 경우에 활용하세요.

제약 사항은 세 가지입니다.

- **Ampere에서는 정수만 지원합니다.** `f32` 변형은 Blackwell이 필요합니다. 다음 절을 참조하세요. `f64`에는 `redux.sync`가 없으므로 어디서나 버터플라이를 유지합니다.
- **워프 전체가 수렴해야 합니다.** 활성 레인 32개 모두 같은 마스크로 같은 호출에 도달해야 합니다. 셔플 축약에도 같은 조건이 필요합니다. 서브워프 타일과 짧은 꼬리 워프는 셔플 경로를 계속 사용합니다(꼬리 처리는 `reduce_sum_f32_partial` 등 참조).
- **`sm_80` 미만에서는 어셈블되지 않습니다.** `redux_sum` 예제처럼 `ctx.compute_capability()`로 호출을 제한하거나 빌드를 Ampere 이상으로 지정하세요.

현재 `warp_reduce`는 이 형식을 자동 선택하지 않습니다. 자동 선택에는 장치 코드가 대상 아키텍처를 알아야 하며, 이 부분은 [#811](https://github.com/NVlabs/cuda-oxide/issues/811)에서 아직 논의 중입니다.

### Blackwell 부동소수점: `f32`용 `redux.sync`

`redux.sync`에는 `f32` 최솟값/최댓값 형식도 추가되었으며 함수 여덟 개로 제공합니다. 각 함수는 정수 함수처럼 `(mask: u32, value: f32)`를 받고 `f32`를 반환합니다.

| 함수 | 축약 연산 |
|:---|:---|
| `warp::redux_sync_min_f32` / `warp::redux_sync_max_f32` | 최솟값 / 최댓값, NaN 입력 무시 |
| `warp::redux_sync_min_abs_f32` / `warp::redux_sync_max_abs_f32` | 절댓값(`.abs`)에 대한 최솟값 / 최댓값 |
| `warp::redux_sync_min_nan_f32` / `warp::redux_sync_max_nan_f32` | NaN 전파(`.NaN`) |
| `warp::redux_sync_min_abs_nan_f32` / `warp::redux_sync_max_abs_nan_f32` | 두 수식자 모두 적용 |

정수 계열과 달리 연산마다 함수 하나가 있는 것은 아닙니다. `.abs`와 `.NaN`은 서로 독립된 명령 수식자이며 각 조합이 별도 진입점입니다.

`.NaN` 형식은 NaN을 잃으면 안 되는 축약에 중요합니다. 기본 형식과 `.abs` 형식은 NaN 입력을 **무시**합니다. `.NaN` 형식은 어느 레인에서든 NaN이 들어오면 결과에 **전파**합니다. 두 방식의 차이는 레인에 NaN이 있을 때만 나타나므로 용도에 맞게 선택하세요.

워프에서 한 레인이 `NaN`이고 일반 값 중 최솟값이 `-1.0`이라고 가정해 보겠습니다.

- `redux_sync_min_f32(mask, v)`는 NaN을 무시해 `-1.0`을 반환합니다.
- `redux_sync_min_nan_f32(mask, v)`는 NaN을 전파해 `NaN`을 반환합니다.

**Blackwell 전용이며 “sm_100 이상”보다 범위가 좁습니다.** `f32` 형식은 PTX 8.6이 필요하며 `sm_100a`, `sm_100f`, `sm_103a`, `sm_103f`에서만 사용할 수 있습니다. `sm_110a`와 `sm_120a`는 지원하지 않으므로 소비자용 Blackwell GPU는 해당되지 않습니다. 네 대상 중 하나로 빌드하지 않으면 컴파일에 실패합니다.

동작 예제는 `redux_f32`입니다. `sm_100a` 대상으로 빌드하므로 CI는 실행하지 않고 컴파일 및 생성된 PTX를 검사합니다. `main`은 장치가 `sm_100`으로 보고될 때만 동작하고 그 외에는 건너뜁니다.

---

## 브로드캐스트

레인 0의 값을 모든 레인에 브로드캐스트하는 데 셔플 하나면 충분합니다.

```rust
let leader_val = warp::shuffle_f32(my_val, 0);
```

값의 원본은 어느 레인이든 될 수 있습니다. 이 방식은 “레인 0이 공유 메모리에 쓰고, 동기화한 뒤, 모든 레인이 읽는” 패턴을 셔플 명령 하나로 대체합니다. 이전에는 세 동작이 필요했습니다.

---

## 포함형 접두 합 (scan)

**포함형 스캔**은 누적 합을 계산합니다. 레인 `i`는 레인 0부터 `i`까지의 합을 보유합니다. 이 패턴은 `shuffle_up`을 사용합니다.

```rust
fn warp_inclusive_scan(mut val: f32) -> f32 {
    let mut offset = 1u32;
    while offset < 32 {
        let n = warp::shuffle_up_f32(val, offset);
        if warp::lane_id() >= offset {
            val += n;
        }
        offset *= 2;
    }
    val
}
```

다섯 단계를 거치면 각 레인이 자신의 값을 포함한 접두 합을 보유합니다. 스트림 압축, 히스토그램 구축, 병렬 스캔 알고리즘의 기본 구성 요소입니다.

---

## 투표: 워프 전체 조건 연산

투표 연산을 사용하면 워프가 불리언 조건을 함께 평가할 수 있습니다.

| 함수 | 반환값 |
|:---|:---|
| `warp::all(pred)` | 활성 레인의 조건이 *모두* 참이면 `true` |
| `warp::any(pred)` | 활성 레인 중 하나라도 조건이 참이면 `true` |
| `warp::ballot(pred)` | 레인 `i`의 조건이 참이면 비트 `i`가 설정되는 `u32` 비트마스크 |
| `warp::popc(pred)` | 조건이 참인 활성 레인의 개수 |

### Ballot을 사용한 필터링

조건을 통과한 원소만 남겨 배열을 압축하는 패턴이 흔히 사용됩니다. `ballot`과 `popc`를 사용하면 개수와 레인별 쓰기 오프셋을 구할 수 있습니다.

```rust
use cuda_device::{DisjointSlice, kernel, thread, warp};

#[kernel]
pub fn compact_positive(
    input: &[f32],
    mut output: DisjointSlice<f32>,
    mut count: DisjointSlice<u32>,
) {
    let idx = thread::index_1d();
    let val = input[idx.get()];
    let is_positive = val > 0.0;

    let mask = warp::ballot(is_positive);
    let lane = warp::lane_id();

    // Count bits below this lane to get the write position
    let offset = (mask & ((1u32 << lane) - 1)).count_ones();

    if is_positive {
        unsafe {
            *output.get_unchecked_mut(offset as usize) = val;
        }
    }

    // Lane 0 records total count for this warp
    if lane == 0 {
        unsafe {
            *count.get_unchecked_mut(warp::warp_id() as usize) = mask.count_ones();
        }
    }
}
```

`ballot` 마스크는 워프 전체의 조건 결과를 레지스터 하나에 인코딩합니다. 통신이나 공유 메모리 없이 하드웨어가 한 사이클에 계산합니다.

---

## 워프 primitive와 공유 메모리 선택

| 작업 | 워프 셔플 | 공유 메모리 |
|:---|:---|:---|
| 값 32개 축약 | 셔플 5회, 약 5사이클 | 로드 5회 + 동기화 5회, 약 50사이클 |
| 값 256개 축약 | 셔플 + 동기화 1회 + 셔플 | 트리 축약, 동기화 약 10회 |
| 스텐실 (이웃 접근) | `shuffle_up` / `shuffle_down` | 2차원 스텐실에 적합 |
| 다른 워프에서 볼 데이터 | 불가능 | 필요 |
| 무작위 접근 패턴 | 지원하지 않음 | 자유롭게 인덱싱 |
| 워프 범위를 넘어 유지 | 해당 없음 | 블록 수명 동안 유지 |

일반적인 기준은 다음과 같습니다. 데이터가 워프 하나(원소 32개)에 들어가고 접근 패턴이 규칙적이면 셔플이 더 빠릅니다. 워프 간 통신이나 더 큰 데이터 집합, 무작위 접근이 필요하면 공유 메모리를 사용하세요.

---

## 전체 예제: 워프 수준 내적

셔플과 투표를 함께 사용해 워프 축약으로 두 벡터의 내적을 계산하는 커널입니다.

```rust
use cuda_device::{DisjointSlice, kernel, thread, warp};

#[kernel]
pub fn warp_dot_product(
    a: &[f32],
    b: &[f32],
    n: u32,
    mut result: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();

    // Each thread computes one element of the pointwise product
    let product = if idx.get() < n as usize {
        a[idx.get()] * b[idx.get()]
    } else {
        0.0f32
    };

    // Warp-level reduction
    let mut sum = product;
    sum += warp::shuffle_xor_f32(sum, 16);
    sum += warp::shuffle_xor_f32(sum, 8);
    sum += warp::shuffle_xor_f32(sum, 4);
    sum += warp::shuffle_xor_f32(sum, 2);
    sum += warp::shuffle_xor_f32(sum, 1);

    // Lane 0 of each warp writes its partial sum
    if warp::lane_id() == 0 {
        unsafe {
            *result.get_unchecked_mut(warp::warp_id() as usize) = sum;
        }
    }
}
```

전체 내적을 계산하려면 두 번째 패스를 실행해 워프별 결과를 다시 축약해야 합니다. 워프 커널을 한 번 더 실행하거나 원자 연산을 사용할 수 있습니다. 첫 패스에서 레지스터 셔플만으로 대부분의 계산을 처리합니다.

:::{seealso}
- 워프보다 큰 작업의 블록 전체 대응 방식은 [공유 메모리와 동기화](../advanced/shared-memory-and-synchronization.md)를 참조하세요.
- 이 패턴에 데이터를 공급하는 전역-공유 메모리 이동을 가속하는 하드웨어는 [텐서 메모리 가속기](tensor-memory-accelerator.md)를 참조하세요.
:::

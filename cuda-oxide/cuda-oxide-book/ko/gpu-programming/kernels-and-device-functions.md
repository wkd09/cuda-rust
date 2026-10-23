# 커널과 디바이스 함수

**커널**은 GPU에서 실행되는 함수로, 호스트가 수천 개 스레드에 걸쳐 실행시키는 진입점입니다. **디바이스 함수**는 GPU에서 실행되는 헬퍼 함수이며 다른 디바이스 함수나 커널에서만 호출할 수 있고 호스트에서는 호출할 수 없습니다. 이 장에서는 두 함수 유형과 디바이스 코드에서 지원되는 Rust 패턴을 설명합니다.

:::{seealso}
커널과 디바이스 함수에 대한 공식 CUDA C++ 설명은 [CUDA 프로그래밍 가이드 — 커널](https://docs.nvidia.com/cuda/cuda-programming-guide/#kernels)을 참조하세요.
:::

## `#[kernel]` — GPU 진입점

함수에 `#[kernel]`을 표시하면 cuda-oxide가 GPU 진입점으로 컴파일합니다. 함수 반환형은 `()`이어야 합니다. 커널은 값을 반환하는 대신 출력 버퍼에 기록해 결과를 전달합니다.

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

내부적으로 `#[kernel]`은 다음 세 가지 작업을 합니다.

1. 컴파일러 수집기가 디바이스 진입점으로 식별하도록 함수를 예약된 `cuda_oxide_kernel_<hash>_<name>` 네임스페이스로 **이름 변경**합니다. 정확한 접두사는 워크스페이스 내부 `reserved-oxide-symbols` 크레이트에서 관리하며 `<hash>` 접미사는 사용자 코드가 이름을 예측하지 못하도록 합니다.
2. 생성된 PTX에서 심볼 이름을 보존하도록 **`#[no_mangle]`을 추가**합니다.
3. 타입 지정 실행 코드가 올바른 PTX 진입점을 찾도록 **호스트 조회 메타데이터를 생성**합니다. 제네릭 커널에는 `scale_ptx_name::<f32>()` 같은 읽기 쉬운 헬퍼도 생성되며 마커 타입은 내부 구현에 사용됩니다.

생성된 PTX에서 커널은 GPU의 `main`에 해당하는 `.entry` 지시문이 됩니다.

```text
.entry vecadd(.param .u64 a, .param .u64 a_len, ...) { ... }
```

### 매개변수 제약

커널 매개변수는 **인자 스칼라화**를 거쳐 호스트/디바이스 ABI 경계를 넘습니다([메모리와 데이터 이동](memory-and-data-movement.md) 장에서 설명). 주요 규칙은 다음과 같습니다.

- **슬라이스**(`&[T]`, `DisjointSlice<T>`)는 포인터와 길이 쌍으로 전달됩니다.
- **스칼라**(`u32`, `f32` 등)는 직접 전달됩니다.
- **값으로 전달되는 구조체와 클로저**는 하나의 byval `.param`으로 전달됩니다. 디바이스 간 내부 호출에서는 필드별 평탄화를 적용하지만, 커널 경계에서는 호스트 실행기가 전달하는 패킷 슬롯 하나에 맞춰 집계값 전체를 하나의 값으로 받습니다.
- **힙 할당 타입**(`Vec`, `String`, `Box`)은 사용할 수 없습니다. 컴파일러가 `alloc` 크레이트를 허용해도 현재 디바이스 측 `#[global_allocator]`는 설정되어 있지 않습니다. 설정하더라도 디바이스 `malloc`은 매우 느립니다.

### 그리드 상수 매개변수

실행 시 전달되는 읽기 전용 값의 주소가 커널에 필요하다면 불변 참조에 `#[grid_constant]`를 사용하세요.

```rust
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Descriptor { bytes: [u8; 128] }

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn consume(#[grid_constant] descriptor: &Descriptor) {
        // Pass descriptor's address to a device helper or a TMA operation.
    }
}
```

생성된 호스트 실행기는 `Descriptor`를 값으로 받습니다. 드라이버는 128바이트를 커널 매개변수 저장 공간에 복사하며, 각 디바이스 스레드는 실행이 진행되는 동안 같은 저장 공간을 빌립니다.

```text
host Descriptor -> one by-value launch argument -> shared read-only device &Descriptor
```

주소를 가져와도 스레드별 디스크립터 복사본은 만들어지지 않습니다. 이는 TMA 텐서 맵에 유용합니다. 별도의 디바이스 할당과 업로드 없이 디스크립터를 실행 인자와 함께 전달할 수 있습니다. 특정 성능 향상을 보장하지는 않으므로 애플리케이션에 중요한 커널과 실행 작업을 측정하세요. 기존 `tma_copy` 예제에서 실제 텐서 맵을 사용한 방식을 볼 수 있습니다.

가리키는 대상은 크기가 알려져 있고 0바이트가 아니며 내부 가변성이 없어야 합니다(`Cell`과 원자 타입을 포함한 `UnsafeCell`은 거부됩니다). 외부 참조의 수명은 생략하거나 `'_`이어야 합니다. 이름이 있는 수명이나 `'static`은 실행이 끝나면 사라지는 저장 공간을 표현할 수 없습니다. 실행 뒤에는 매개변수를 수정하거나 주소를 사용하지 마세요. 디스크립터 *내부*에 저장된 포인터도 GPU가 접근할 수 있는 메모리를 가리켜야 합니다. 생성된 타입 지정 실행기는 다른 값 전달 인자처럼 해당 값에 `Copy`를 요구합니다. 일반적인 값 전달 레이아웃 제한도 적용됩니다. 디바이스 공유 메모리 포인터는 지원 백엔드별 저장 폭이 달라 거부되지만 일반 제네릭 또는 전역 포인터에는 이 제한이 없습니다.

그리드 상수 매개변수용 호스트 실행 메서드는 준비 실행 및 비동기 메서드를 포함해 항상 `unsafe`입니다. 디바이스 커널 함수가 안전해도 마찬가지입니다. `Copy`와 읽기 전용 매개변수 저장 공간만으로 값 안의 참조나 포인터가 GPU에서 유효하다고 증명할 수 없습니다. 실행 시 호출자는 커널이 접근하는 모든 할당이 디바이스에서 접근 가능하고 올바르게 정렬 및 초기화되어 있으며 GPU 작업 완료까지 살아 있고 Rust의 별칭 및 동기화 규칙을 만족하는지 보장해야 합니다. 준비 실행은 형상과 리소스를 확인하지만 이런 메모리 속성은 증명하지 않습니다. 현재 타입 제약은 데이터만 포함한 값과 포인터 포함 값을 구별하지 않으므로 데이터 전용 페이로드에도 같은 요구가 적용됩니다.

수동 unsafe 실행에서는 가리키는 값 전체를 Rust 크기와 정렬에 맞춰 인자 하나로 전달하세요. 8바이트 디바이스 포인터를 전달하면 ABI와 일치하지 않습니다. 일반 주석 없는 포인터 매개변수는 기존 포인터 ABI를 유지하므로 전역 메모리에 있거나 그곳에서 갱신할 디스크립터에 사용하세요. 그리드 상수 매개변수에는 컴퓨트 기능 7.0 이상이 필요합니다. LLVM NVPTX와 libNVVM 경로 모두 값 전달 크기와 정렬을 보존합니다.

## 디바이스 헬퍼 함수

GPU 코드를 모두 커널 본문에 넣을 필요는 없습니다. 로직을 헬퍼 함수로 분리하면 컴파일러가 해당 함수도 GPU용으로 컴파일합니다.

### 자동 탐색되는 헬퍼

가장 간단한 방법은 일반 Rust 함수를 작성해 커널에서 호출하는 것입니다. 컴파일러의 **수집기**는 각 `#[kernel]` 진입점부터 호출 그래프를 따라가며 도달 가능한 모든 함수를 GPU용으로 자동 컴파일하므로 별도 주석이 필요 없습니다.

```rust
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

#[kernel]
pub fn apply_clamp(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = clamp(input[idx.get()], 0.0, 1.0);
    }
}
```

`clamp` 함수는 PTX `.func`(디바이스 함수)로 컴파일되며 보통 컴파일러가 인라인하므로 호출 오버헤드가 없습니다.

### `#[device]`가 필요한 경우

자동 탐색만으로 충분하지 않은 다음 세 가지 경우에는 `#[device]` 속성이 필요합니다.

| 상황                         | `#[device]`가 필요한 이유                                                             |
|:---------------------------------|:--------------------------------------------------------------------------------------|
| **독립 디바이스 라이브러리**  | 크레이트에 `#[kernel]`이 없어 수집기가 탐색할 진입점이 없습니다.         |
| **크레이트 간 디바이스 함수** | 함수가 커널과 다른 크레이트에 있습니다.                                  |
| **디바이스 FFI**                   | LTOIR로 CUDA C++와 링크하도록 함수가 `#[device] extern "C"`로 노출됩니다. |

```rust
use cuda_device::device;

#[device]
pub fn magnitude(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}
```

### `#[kernel]`과 `#[device]` 비교

| 기능                  | `#[kernel]`             | `#[device]`                         | 자동 탐색        |
|:-------------------------|:------------------------|:------------------------------------|:-----------------------|
| PTX 지시문            | `.entry`                | `.func`                             | `.func`(또는 인라인)   |
| 호스트에서 실행 가능     | 예, 타입 지정 모듈 사용   | 아니요                                  | 아니요                     |
| 값을 반환할 수 있음       | 아니요(`()`이어야 함)       | 예                                 | 예                    |
| 디바이스 코드에서 호출 가능| 예                     | 예                                 | 예                    |
| 주석 필요      | 항상                  | 독립/크레이트 간/FFI에서만 | 불필요                  |

### 헬퍼 인라이닝

헬퍼의 `#[inline]` 요청은 호스트 백엔드와 같은 방식으로 GPU에 적용됩니다. `#[inline]`은 LLVM `inlinehint`, `#[inline(always)]`는 `alwaysinline`, `#[inline(never)]`는 `noinline`이 됩니다. 따라서 `never`로 표시한 헬퍼는 PTX에서 별도 `.func`로 남고 실제 `call.uni`로 호출됩니다. 단, 자체 타입 또는 const 매개변수가 있는 헬퍼에 `#[device]`가 추가하는 `#[inline(never)]`는 호스트 측 수집 경계이지 디바이스 인라이닝 의도가 아니므로 전달되지 않습니다. const 제네릭 인자는 계속 커널에 접힙니다.

```{figure} images/inline-intent-mapping.svg
:align: center
:width: 100%

How an inline request reaches the GPU. Left, one helper of each kind from the
`inline_intent` example. Middle, the request recorded on `mir.func` as a typed
`inline_intent` attribute and lowered to the LLVM function attribute in one
place. Right, what opt and llc make of it: `call.uni` for the `noinline`
helpers, no function left for the `alwaysinline` one, and the const-generic
factor folded to an immediate. Before this mapping only `alwaysinline`
survived, and LLVM inlined a `never` helper anyway.
```

## GPU에서 사용할 수 있는 Rust 기능

cuda-oxide는 `rustc`를 통해 표준 Rust를 컴파일하므로 부분 언어가 아닙니다. 다만 GPU 코드는 디바이스 측 힙 할당자가 설정되지 않은 `no_std` 환경에서 실행되므로 현재 사용할 수 없는 기능도 있습니다. 현재 지원 현황은 다음과 같습니다.

### 지원

| 기능                                                | 설명                                          |
|:-------------------------------------------------------|:-----------------------------------------------|
| 원시 타입(`u8`..`u64`, `f32`, `f64`, `bool`)    | 완전히 지원                                   |
| 구조체와 튜플                                     | ABI 경계에서 분해                     |
| 열거형(`Option<T>`, `Result<T,E>`, 사용자 정의)             | `match` 포함                              |
| `match` / `if` / `if let`                              | 다중 분기                            |
| `for` 및 `while` 루프                          | 범위 및 반복자 기반                 |
| 반복자(`.iter()`, `.enumerate()`)                  | MIR을 통해 구문 변환                          |
| `break`와 `continue`                                 | 루프 내부에서 사용                                   |
| 배열(`[T; N]`)                                      | 읽기, 쓰기, 인덱싱                          |
| 슬라이스(`&[T]`)                                        | 읽기 전용. 가변 쓰기에는 `DisjointSlice` 사용  |
| 클로저(디바이스 코드 안)                          | 일반 Rust 의미 체계                          |
| 제네릭 함수                                      | 호출 위치마다 단형화                    |
| `unsafe` 블록과 원시 포인터                       | 고급 패턴에서 사용                          |

### 지원하지 않는 기능

| 기능                           | 이유                                                              | 대안                          |
|:----------------------------------|:--------------------------------------------------------------------|:-------------------------------------|
| `String`, `Vec`, `Box`            | 힙 할당자가 필요함(현재 디바이스 측 `#[global_allocator]` 없음) | 고정 크기 배열 또는 슬라이스 사용      |
| `format!`, `println!`             | 포매팅 기능과 입출력이 필요함                                  | `gpu_printf!` 사용                    |
| `std` 입출력, 네트워킹, 파일 시스템 | GPU에는 OS가 없음                                                        | 버퍼로 통신              |
| 트레잇 객체(`dyn Trait`)       | vtable 디스패치가 필요함                                             | 제네릭 사용(단형화)         |

:::{tip}
If you accidentally use an unsupported feature, the compiler will produce a
clear error: `"CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE"` with a list of
allowed crates (`core`, `alloc`, `cuda_device`, and your local crate).
:::

## 조건부 컴파일

디바이스 코드에서도 `#[cfg(...)]`는 다른 Rust 코드와 동일하게 동작합니다. 부족한 점은 조건으로 삼을 값입니다. 컴파일러가 타깃에서 파생한 `cfg`를 제공하지 않으므로 커널은 대상 아키텍처를 알 수 없습니다. 아키텍처 요구 사항은 문서 주석에 기록하고 호출자가 적용합니다. 따라서 `redux.sync`가 필요한 예제는 커널을 특수화하는 대신 호스트에서 `ctx.compute_capability()`를 확인하고 지원되지 않으면 건너뜁니다.

직접 조건을 제공하려면 `--device-cfg NAME`을 사용합니다. 반복 지정할 수 있으며 각 항목은 빌드 rustflags의 `--cfg NAME`이 됩니다.

```bash
# 크레이트 디렉터리에서 실행하세요. 예제 이름은 지정하지 않습니다(아래 참조).
cargo oxide build --device-cfg ampere_up
```

```rust
// 타깃이 지원하면 명령 하나를 사용하고, 그 외에는 셔플 트리를 사용합니다.
#[cfg(ampere_up)]
let total = warp::redux_sync_add(u32::MAX, value);

#[cfg(not(ampere_up))]
let total = {
    let mut acc = value;
    let mut offset = 16;
    while offset > 0 {
        acc = acc.wrapping_add(warp::shuffle_xor(acc, offset));
        offset /= 2;
    }
    acc
};
```

사용하기 전에 알아둘 점이 네 가지 있습니다.

**디바이스 코드에만 적용되지 않습니다.** 이 플래그는 rustflag로 전달되므로 호스트 코드를 포함해 해당 빌드에서 cargo가 컴파일하는 모든 크레이트에서 `cfg`를 볼 수 있습니다.

**`build`가 전달 모드로 전환됩니다.** 이 플래그를 지정하면 `build`는 예제 이름을 받지 않으므로 `cargo oxide build my_example --device-cfg ampere_up`는 거부됩니다. 크레이트 디렉터리에서 실행하세요. (`test`는 이미 항상 전달 모드입니다.) 조건을 매니페스트에 둘 수 있다면 Cargo 기능(`#[cfg(feature = "ampere_up")]`)으로 예제 이름 실행을 유지하면서 같은 효과를 얻을 수 있습니다. Cargo.toml을 수정하지 않고 `cfg`를 주입해야 할 때 `--device-cfg`를 사용하세요.

**선언되지 않은 `cfg` 이름에 대해 rustc가 경고합니다.** `unexpected_cfgs` 린트는 각 `#[cfg(...)]`를 선언 집합과 비교합니다. 주입된 `--cfg ampere_up`은 집합에 없으므로 사용할 때마다 `unexpected cfg condition name` 경고가 출력됩니다. 이를 없애려면 커널 크레이트 매니페스트에 선언하세요.

```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ["cfg(ampere_up)"] }
```

**`--arch`와 연결되지 않습니다.** `cfg` 이름이 아키텍처를 뜻한다면 두 옵션을 일치시키는 것은 사용자 책임입니다. 해당 `--arch` 없이 `--device-cfg ampere_up`만 전달하면 선택한 타깃에 특수 경로를 빌드하며 어셈블러가 PTX를 거부할 수 있습니다.

(loop-unrolling)=

## 루프 언롤링

`#[kernel]` 또는 `#[device]` 함수에서 반복 횟수를 컴파일 시점에 아는 루프 바로 앞에 `#[unroll]`을 붙이세요. 컴파일러가 루프를 제거하고 본문을 복제해 배치하도록 요청합니다.

```rust
#[kernel]
pub fn sum_four(mut out: DisjointSlice<u32>) {
    let tid = thread::index_1d();
    if let Some(out_elem) = out.get_mut(tid) {
        let mut sum = 0;
        let mut i = 0;
        #[unroll]
        while i < 4 {
            sum += i;
            i += 1;
        }
        *out_elem = sum;
    }
}
```

현재 패스는 `isize`를 포함한 정수 카운터의 명시적 반복 횟수 `while` 루프와 범위 `for` 루프(`for i in a..b`, `(a..b).rev()`)를 인식합니다. `a..=b`나 `.step_by(k)` 같은 다른 반복자 루프는 아직 인식하지 않습니다.

반복 횟수를 런타임에만 알 수 있다면 `N >= 2`인 `#[unroll(N)]`을 사용하세요. 루프는 한 번에 `N`회 작업을 수행합니다. 남은 반복은 나머지 루프에서 처리하므로 `n`이 `N`의 배수일 필요는 없습니다.

```rust
let mut i = 0;
#[unroll(4)]
while i < n {
    process(i);
    i += 1;
}
```

주석이 붙은 루프 안에 다른 루프가 있어도 됩니다. 주석이 붙은 루프만 언롤되며 내부 루프는 그대로 복사되어 루프로 남습니다. 내부 루프도 언롤하려면 별도 주석을 추가하세요.

`continue` 경로가 여러 개인 루프를 지원합니다. 전체 `#[unroll]`은 `break` 경로와 종료 지점이 여러 개인 루프도 보존합니다.

부분 언롤 `#[unroll(N)]`은 현재 루프 조건만 종료 경로인 경우에 사용할 수 있습니다. `break`나 다른 종료가 있으면 컴파일러가 경고하고 해당 루프를 언롤하지 않습니다.

부분 언롤에는 증가하는 카운터 루프가 필요합니다. 증가폭은 양수이고 `<` 또는 `<=`를 사용하며 루프 안에서 바뀌지 않는 상한과 비교해야 합니다. 지원되지 않는 요청은 컴파일러가 경고하고 언롤하지 않습니다. 언롤되지 않은 루프의 경고는 `-->` 줄에 `file:line:column` 형식으로 `#[unroll]` 위치를 표시하므로 주석이 여럿이어도 어떤 루프가 유지됐는지 알 수 있습니다.

생성 코드 크기를 제한하기 위해 주석 하나가 만드는 본문 복사본은 최대 1,024개, 복제 기본 블록은 8,192개, 복제 작업은 65,536개입니다. 이를 넘으면 경고 후 언롤하지 않습니다. 전체 변수 디버그 빌드도 루프 변수를 SSA 대신 메모리에 보존하므로 언롤을 생략합니다.

언롤링은 분기를 줄이고 최적화 기회를 늘리는 대신 생성 코드가 커집니다. 작거나 성능이 중요한 루프에 적용하고 결과를 측정하세요.

:::{seealso}
스테이지 인덱스 peephole을 포함해 주석이 붙은 루프를 컴파일러가 분석하고 다시 쓰는 방법은 [컴파일러 최적화](../compiler/compiler-optimizations.md)를 참조하세요.
:::

## `#[launch_bounds]` — 점유율 힌트

`#[launch_bounds]` 속성은 블록마다 실행하려는 스레드 수를 컴파일러에 알립니다. PTX 어셈블러의 레지스터 할당을 개선하고 점유율을 높일 수 있습니다.

```rust
#[kernel]
#[launch_bounds(256, 2)]
pub fn optimized_kernel(mut out: DisjointSlice<f32>) {
    // ...
}
```

| 매개변수      | 필수 | PTX 지시문   | 설명                      |
|:---------------|:---------|:----------------|:---------------------------------|
| `max_threads`  | 예      | `.maxntid`      | 블록당 최대 스레드 수        |
| `min_blocks`   | 아니요       | `.minnctapersm` | SM당 최소 동시 블록 수 |

생성되는 PTX에는 다음 지시문이 포함됩니다.

```text
.entry optimized_kernel .maxntid 256, 1, 1 .minnctapersm 2 { ... }
```

`.maxntid`는 `x * y * z`의 곱을 제한하므로 스레드 256개 제한은 `(256, 1, 1)`, `(16, 16, 1)`, `(4, 8, 8)`을 모두 허용합니다. 정확한 형상이 필요하면 `#[launch_contract(block = (x, y, z))]`를 추가하세요. `.maxntid` 대신 `.reqntid`를 출력하고 드라이버가 축별로 검사합니다.

```text
.entry exact_kernel .reqntid 256, 1, 1 .minnctapersm 2 { ... }
```

두 지시문은 함께 사용할 수 없습니다. ptxas는 둘 다 선언한 진입점을 거부하므로 계약이 적용된 커널은 `.reqntid`만 출력합니다. `.minnctapersm`은 점유율 힌트이며 둘 중 어느 지시문과도 함께 쓸 수 있습니다.

:::{tip}
`#[launch_bounds]`는 `#[kernel]` **뒤에** 와야 합니다.

```rust
#[kernel]
#[launch_bounds(256, 2)]   // correct
pub fn my_kernel(...) { }
```

:::

### 제한으로 레지스터 스필이 발생하는 경우

실행 제한은 결국 스레드당 레지스터 수의 상한입니다. ptxas에 상주 스레드를 더 많이 약속하면 각 스레드가 사용할 레지스터가 줄어듭니다. 커널이 허용량보다 많은 레지스터를 필요로 하면 ptxas는 실패하는 대신 초과분을 로컬 메모리로 **스필**합니다. 스필된 값에 접근할 때마다 제한으로 높인 점유율의 비용을 다시 치르게 됩니다.

빌드가 성공하므로 놓치기 쉽습니다. cuda-oxide는 대신 커널 정의 위치에서 경고합니다.

```text
warning: kernel `cuda_oxide_kernel_a1b2c3d4_my_kernel` compiled with
         `#[launch_bounds(256, 2)]` and spills registers
  = note: ptxas reports 96 bytes spill stores and 96 bytes spill loads
  = note: ptxas allocated 40 registers per thread
  = help: relax `min_blocks_per_sm` or reduce register pressure
```

(바이트 수와 레지스터 개수는 ptxas가 해당 커널에 대해 보고한 값입니다.) `#[launch_bounds]`가 있는 커널만 검사합니다. 제한이 없으면 레지스터를 희생하며 지켜야 할 약속이 없습니다. 이 경고는 네이티브 cubin을 생성하는 빌드(`--materialize-cubin` 또는 cubin을 포함하는 빌드)에서만 나옵니다. PTX만 빌드하면 ptxas가 실행되지 않아 리소스 보고서가 없습니다.

일반적인 대응은 `max_threads`를 높이거나 `min_blocks`를 낮추거나 완화하거나 커널의 레지스터 사용량을 줄이는 것입니다. 스필이 항상 나쁜 것은 아닙니다. 점유율이 높은 상태에서 조금 스필하는 커널이 점유율이 낮은 상태에서 스필하지 않는 커널보다 빠를 수 있습니다. 이 경고는 그 선택을 우연이 아닌 의도에 따라 하도록 알려줍니다.

:::{warning}
`#[allow(...)]`로 이 경고를 **숨길 수 없습니다**. 린트가 아니라 원시 위치 진단이므로 억제 속성이 적용되지 않습니다. 스필을 측정하고 받아들이기로 한 빌드에서는 환경 변수로 억제할 수 있습니다.

```bash
CUDA_OXIDE_NO_SPILL_WARN=1 cargo oxide build my_kernels --materialize-cubin
```

:::

## 수집기 — 디바이스 코드 검색 방식

`cargo oxide`로 빌드하면 `rustc-codegen-cuda` 백엔드가 **수집기** 패스를 실행해 GPU용으로 컴파일할 함수를 결정합니다.

1. 모든 컴파일 단위에서 예약된 `cuda_oxide_kernel_<hash>_` 네임스페이스(`#[kernel]`이 생성)를 사용하는 함수를 찾습니다.
2. 각 커널에서 **호출 그래프를 탐색**해 전이적으로 도달 가능한 함수를 모두 모읍니다.
3. 호출된 함수를 허용 크레이트 목록에 따라 **필터링**합니다.

| 크레이트            | 허용 | 이유                                                                                        |
|:-----------------|:--------|:-------------------------------------------------------------------------------------------|
| 현재 로컬 크레이트 | 예     | 커널과 헬퍼 코드                                                                |
| `cuda_device`    | 예     | GPU 내장 기능(스레드, 워프, 공유 메모리)                                             |
| `core`           | 예     | `no_std` Rust 핵심 라이브러리                                                                 |
| `std`            | 아니요      | GPU에서 사용할 수 없는 OS 기능이 필요합니다.                                                |
| `alloc`          | 허용 | 수집기는 통과하지만 디바이스 측 할당자가 아직 연결되지 않아 현재는 링크 오류가 발생합니다. |

수집기가 허용되지 않은 크레이트 호출을 발견하면 잘못된 PTX를 생성하는 대신 컴파일 오류를 보고합니다.

```{figure} images/collector-traversal.svg
:align: center
:width: 100%

디바이스 코드 수집기는 `#[kernel]` 진입점부터 호출 그래프를 따라 도달 가능한 디바이스 함수를 찾고, 각 호출을 허용 크레이트(로컬 크레이트, cuda_device, core) 목록에 따라 필터링합니다. 출력은 `.entry`와 `.func` 지시문이 포함된 PTX 모듈입니다.
```

## `no_std` 및 패닉 동작

디바이스 코드는 암시적으로 `#![no_std]` 환경에서 실행됩니다. 직접 속성을 추가할 필요 없이 컴파일러 백엔드가 처리합니다.

**패닉 동작:** MIR의 모든 언와인드 경로는 도달 불가로 취급됩니다. 런타임에 패닉이 실제 발생하면(예: 배열 범위 검사 실패) GPU가 **트랩 명령**을 실행하고 호스트는 `CUDA_ERROR_ILLEGAL_INSTRUCTION`을 받습니다. 의미상 `panic=abort`와 같지만 특별한 컴파일러 플래그는 필요하지 않습니다.

실제로는 다음과 같이 동작합니다.

- `unwrap()`과 `expect()`는 동작하지만 `None`/`Err`에서 GPU 트랩을 발생시킵니다.
- `assert!`와 `debug_assert!`는 동작하지만 실패하면 트랩을 발생시킵니다.
- `panic!("message")`와 `panic!("{}", value)` 모두 컴파일됩니다. 패닉 경로는 트랩으로 하위 변환되고 **메시지는 버려집니다**. 메시지를 출력하는 패닉 런타임이 없고 호출이 제거되면 메시지를 만들 문장도 필요하지 않기 때문입니다. 값을 확인하려면 검사 전에 `gpu_printf!`를 사용하고, 명시적 검사에는 `gpu_assert!`를 사용하세요.

실제 본문이 있는 `-> !` 디바이스 함수는 다릅니다. 오류 코드를 기록하고 `loop {}`를 실행하는 경우처럼 호출자에게 돌아오지 않습니다. 컴파일러는 다른 함수처럼 호출하고 뒤에 `unreachable`을 배치하므로 발산 전에 수행한 작업이 유지됩니다. 단, `core::panicking`의 패닉 진입점, 패닉만 있는 `-> !` 본문, 함수 포인터를 통한 호출은 여전히 디바이스 트랩으로 하위 변환됩니다.

```{figure} images/diverging-call-lowering.svg
:align: center
:width: 100%

`-> !` 디바이스 함수 호출의 하위 변환입니다. 왼쪽은 값을 저장한 뒤 반복하는 Rust의 `write_then_diverge` 코드입니다. 가운데는 이전 임포터의 출력으로, 호출이 `trap;`으로 대체되어 저장이 수행되지 않습니다. 오른쪽은 현재 출력으로, 호출 뒤에 `unreachable`이 배치되어 PTX가 값을 저장하고 반복합니다. 패닉 진입점, 패닉만 있는 `-> !` 본문, 함수 포인터 호출은 여전히 트랩으로 하위 변환됩니다.
```

:::{seealso}
커널 실패 진단에 사용하는 `gpu_printf!`, `gpu_assert!`, `cargo oxide debug`는 [오류 처리와 디버깅](error-handling-and-debugging.md) 장에서 설명합니다.
:::

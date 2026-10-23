# 클로저와 제네릭

Rust의 제로 비용 추상화인 제네릭, 클로저, 트레이트 바운드는 GPU에서도 작동합니다. 이는 cuda-oxide의 가장 두드러진 기능 중 하나입니다. 런타임 오버헤드 없이 모든 숫자 타입을 처리하는 제네릭 커널 하나를 작성하거나, 호스트에서 클로저를 전달해 GPU 동작을 사용자 정의할 수 있습니다.

## 제네릭 커널

일반 Rust 함수와 마찬가지로 커널도 타입, const 값, 트레이트 바운드에 대해 제네릭일 수 있습니다. 컴파일러는 각 특수화를 별도의 PTX 진입점으로 모노모피제이션합니다.```rust
use core::ops::Mul;
use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(
    factor: T,
    input: &[T],
    mut out: DisjointSlice<T>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[i] * factor;
    }
}
```

### PTX 이름

모노모피제이션으로 특수화된 항목마다 고유한 PTX 진입점이 생깁니다. 제네릭이 아닌 커널은 일반 함수 이름을 유지합니다. 클로저 제네릭 커널을 포함한 제네릭 커널에는 `_TID_<hex32>` 접미사가 붙습니다. `<hex32>`는 생성된 구체 커널 함수 항목에 대한 rustc의 안정적인 type-id 해시를 소문자 16진수 32자로 표시한 값입니다.

| 인스턴스화 | PTX 진입점 이름 |
|:---|:---|
| 제네릭이 아닌 `vecadd` | `vecadd` |
| `scale::<f32>` | `scale_TID_<hex32>` |
| `scale::<MyType>` | `scale_TID_<hex32>` |
| `tile::<4>` | `tile_TID_<hex32>` |
| `tile::<8>` | `tile_TID_<hex32>` |
| 클로저인 `map::<f32, _>` | `map_TID_<hex32>` |

호스트 실행기와 장치 백엔드는 하나의 통합 빌드 안에서 고정된 rustc 도구 체인에 동일한 `FnDef` 해시를 요청하므로 문자열이 바이트 단위로 일치합니다. 함수 항목에는 정의와 순서가 지정된 모든 타입 및 const 인자가 들어갑니다. 따라서 wire 이름은 제네릭 인자 수와 관계없이 고정 길이를 유지합니다. 해시 전에 빌림 수명은 제거되므로 수명 때문에 GPU 코드가 중복 생성되지 않습니다.

이 접미사는 영구 ABI가 아니라 빌드 산출물로 취급하세요. 호스트 코드와 PTX는 cuda-oxide의 고정 rustc 도구 체인으로 함께 빌드해야 합니다.

### 제네릭 커널 실행

실행할 때 생성된 타입 지정 메서드에 타입 매개변수를 지정합니다. 그러면 구체 인스턴스화가 강제되고 로더가 해당 PTX 진입점을 찾을 수 있습니다.```rust
use cuda_core::simt::LaunchConfig;

// SAFETY: config is 1D and both buffers cover every launched index.
unsafe {
    module.scale::<f32>(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        2.0f32,
        &input_dev,
        &mut output_dev,
    )
}
.expect("Launch failed");
```

생성된 메서드가 `scale::<f32>`의 모노모피제이션을 강제하므로 CPU에서 직접 호출하지 않더라도 해당 인스턴스가 컴파일된 PTX에 포함됩니다.

### 컴파일 시점 커널 정책

정책 타입은 제네릭 커널 하나에 적용할 튜닝 선택지를 묶습니다.```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::LaunchConfig1D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};

trait TransformPolicy {
    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum Small {}
impl TransformPolicy for Small {
    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
    const UNROLL: u32 = 2;
}

enum Wide {}
impl TransformPolicy for Wide {
    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
    const UNROLL: u32 = 4;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub fn transform<P: TransformPolicy>(mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        if let Some(value) = output.get_mut(index) {
            let mut step = 0;
            #[unroll(P::UNROLL)]
            while step < 8 {
                *value = value.rotate_left(1);
                step += 1;
            }
        }
    }
}

let launch = module
    .prepare_transform::<Small>(LaunchConfig1D::new(1, 64, 0))
    .expect("valid Small launch");
module
    .transform::<Small>(&stream, &launch, &mut output)
    .expect("launch Small kernel");

let wide_launch = module
    .prepare_transform::<Wide>(LaunchConfig1D::new(1, 256, 0))
    .expect("valid Wide launch");
module
    .transform::<Wide>(&stream, &wide_launch, &mut output)
    .expect("launch Wide kernel");
```

`Small`과 `Wide`는 서로 다른 PTX 커널을 생성합니다. 정책은 커널 인자가 아니며 GPU가 런타임에 정책을 선택하지 않습니다.

정책에는 메타데이터 전용 설명도 넣을 수 있습니다.```rust
use cuda_device::config::{Atom, AtomKind, Block, Global, RowMajor, Shape1, Thread, Tile};

type BlockTile = Tile<Shape1<1024>, RowMajor, Global, Block>;

enum Rotate {}
impl AtomKind for Rotate {}
type ElementOp = Atom<Rotate, Shape1<1>, Thread>;
```

`BlockTile`은 블록이 처리하는 행 우선 전역 메모리 원소 1,024개를 나타냅니다. `ElementOp`는 스레드가 처리하는 원소 하나짜리 연산을 이름 붙이며 `Rotate`의 동작은 커널 라이브러리가 정합니다. 두 타입 모두 메모리를 할당하거나 주소를 계산하거나 연산을 실행하지 않습니다.

`MAX_THREADS`는 최대 스레드 수이지 정확한 블록 크기가 아닙니다. `Small`의 준비된 실행은 블록당 최대 64개, `Wide`는 최대 256개 스레드를 허용합니다. 크기를 정확히 고정해야 하는 경우에만 `#[launch_contract]`에서 `block = (x, y, z)`를 사용하세요. `MIN_BLOCKS`는 컴파일러 점유율 힌트이지 요청하는 그리드 크기가 아닙니다.

`UNROLL = 4`는 루프 반복 네 개를 한 묶음으로 처리합니다. 네 번 실행한 뒤 루프를 끝내는 것이 아니며 나머지 반복도 계속 처리됩니다.

제네릭 정책 표현식에는 현재 Rust의 `generic_const_exprs` 기능이 필요합니다. 계약이 있는 `launch_bounds`에서 사용하는 다른 이름 있는 상수는 모듈 범위에 선언되어야 합니다.

### Const 제네릭 커널

커널 및 장치 함수 진입점에서 const 매개변수를 사용할 수 있습니다.```rust
#[kernel]
pub fn add_value<const VALUE: u32>(mut output: DisjointSlice<u32>) {
    let index = thread::index_1d();
    if let Some(element) = output.get_mut(index) {
        *element += VALUE;
    }
}
```

`add_value::<4>` → `add_value_TID_<hash A>`
`add_value::<8>` → `add_value_TID_<hash B>```text
add_value::<4> -> add_value_TID_<hash A>
add_value::<8> -> add_value_TID_<hash B>
```

생성된 실행 메서드에는 일반적인 Rust turbofish 구문을 사용합니다.```rust
// SAFETY: config matches the kernels' 1D indexing and output bounds.
unsafe {
    module.add_value::<4>(&stream, config, &mut output)?;
    module.add_value::<{ Config::VALUE }>(&stream, config, &mut output)?;
}
```

오래된 편의 문법 `#[kernel(f32, i32)]`은 타입 매개변수 하나만 지원합니다. const 제네릭이나 혼합 제네릭에는 기본 `#[kernel]`을 사용하고 실행 지점에서 특수화하세요.

## 커널 인자로 호스트 클로저 전달

cuda-oxide는 호스트에서 GPU로 클로저를 전달할 수 있습니다. 커널 동작을 함수로 매개변수화하는 `map` 스타일 패턴을 사용할 수 있습니다.```rust
#[kernel]
pub fn map<F: Fn(i32) -> i32>(f: F, input: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[i]);
    }
}
```

클로저를 전달해 실행합니다.```rust
let factor = 3i32;
// SAFETY: config is 1D and both buffers cover every launched index.
unsafe {
    module.map::<_>(&stream, config, move |x| x * factor, &input_dev, &mut output_dev)
}
.expect("Launch failed");
```

### 클로저 인자의 전달 방식

클로저는 캡처한 필드 목록이 아니라 값 하나로 실행기에 전달됩니다. 실행기는 클로저 구조체 전체(모든 캡처 포함)를 드라이버 인자 하나로 넘기고, 커널은 이를 byval `.param` 하나로 받습니다.```text
host          factor = 3i32; cl = move |x| x * factor
              push one driver arg ─► closure struct { factor: i32 }

GPU kernel    .entry map_TID_<hex>(
                .param .align 4 .b8 f[4],    ; one byval closure
                .param .u64 input_ptr,        ; slice still (ptr, len)
                .param .u64 input_len,
                ...
              )
```

슬라이스는 호스트 실행 헬퍼와 PTX 진입점 레이아웃이 공유하는 형태이므로 `(ptr, len)`으로 평탄화됩니다. 커널 경계에서 byval 하나로 전달되는 것은 클로저와 값으로 전달된 사용자 구조체처럼 집합 형태의 값입니다.

캡처가 없는 클로저는 크기가 0인 타입입니다. 백엔드는 `.param`을 완전히 생략하고 호스트 실행기도 이를 건너뛰어 패킷 정렬을 유지합니다.

### `move` 클로저와 참조 클로저

`move` 키워드는 캡처가 GPU로 전달되는 방식을 결정합니다.

### Move 클로저 (기본 권장)```rust
let factor = 3i32;
move |x| x * factor   // `factor` is copied into the closure struct
```

- 클로저 구조체가 캡처를 값으로 보유합니다(`{ factor: i32 }`).
- 커널은 byval 클로저의 일반 필드처럼 `factor`를 읽습니다.
- 실행 후 호스트 변수를 해제할 수 있습니다.
- 특별한 하드웨어 지원 없이 모든 시스템에서 작동합니다.

### 참조 클로저 (HMM)```rust
let factor = 3i32;
|x| x * factor   // closure captures &factor
```

- 클로저 구조체는 `factor`를 가리키는 **호스트 포인터**를 포함합니다(`{ factor: &i32 }`).
- 클로저 전체는 여전히 byval 인자 하나로 전달됩니다. 커널은 **하드웨어 관리 메모리(HMM)**를 통해 호스트 페이지를 접근할 때 이동시키며 포인터를 역참조합니다.
- 커널이 완료될 때까지 호스트 변수를 **계속 살아 있게 유지해야 합니다**.
- HMM 지원이 필요합니다(Turing 이상 GPU, Linux 6.1.24 이상, CUDA 12.2 이상).

### 어떤 방식을 선택할까요?

| 상황 | 선택 |
|:---|:---|
| 작은 스칼라 캡처(숫자, 불리언) | `move` (복사 오버헤드 없음) |
| 큰 구조체 캡처 | 커널이 여러 번 읽으면 `move`, 드물게 접근하면 HMM |
| 프로토타이핑 | 둘 다 가능하지만 `move`가 더 이식성이 높음 |
| 호스트와 장치 사이에 가변 상태 공유 | 참조(HMM). 동기화에 유의하세요. |

:::{tip}
잘 모르겠다면 `move` 클로저를 사용하세요. 이해하기 쉽고 어디서나 작동하며 호스트/장치 메모리 공유로 인한 동기화 위험을 피할 수 있습니다.
:::

## 커널 내부 클로저

장치 코드 안에서 정의하고 호출하는 클로저는 일반 Rust 의미대로 작동합니다. 이미 GPU에 있는 코드이므로 호스트/장치 ABI 경계를 통과하지 않습니다.```rust
#[kernel]
pub fn apply_transform(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    let transform = |x: f32| -> f32 {
        let clamped = if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };
        clamped * clamped
    };

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = transform(input[idx.get()]);
    }
}
```

커널 내부 클로저는 컴파일러가 인라인 처리하므로 오버헤드가 없습니다. 별도 장치 함수를 만들지 않고 커널 내부 로직을 분리할 때 유용합니다.

## 크레이트 간 커널

라이브러리 크레이트에서 커널을 정의하고 바이너리 크레이트에서 실행할 수 있습니다.```rust
// In lib crate `my_kernels`:
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
```

```rust
// In binary crate:
use my_kernels::kernels;

let module = kernels::load(&ctx)?;
// SAFETY: config is 1D and all three buffers cover every launched index.
unsafe { module.vecadd(&stream, config, &a, &b, &mut c) }
    .expect("Launch failed");
```

컴파일러는 `#[kernel]`이 생성하는 마커 트레이트를 통해 크레이트 간 커널을 탐색합니다. 타입이 지정된 모듈이 컴파일 시점에 PTX 이름을 확인하고 로드된 함수 핸들을 캐시합니다.

:::{tip}
크레이트 간 제네릭 커널은 구체 타입이 알려진 **호출 측** 크레이트에서 모노모피제이션됩니다. 따라서 해당 바이너리를 컴파일할 때 PTX가 생성됩니다.
:::

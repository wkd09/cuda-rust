# 커널 실행

커널을 작성하는 것은 절반의 이야기일 뿐이다. 호스트는 디바이스 코드를 로드하고, 실행 그리드를 구성하고, 마샬 알루먼트를 설정하고, 작업을 GPU에 전송해야 한다. 주 cuda-oxide 실행 경로는 이다.`#[cuda_module]`: 생성된 디바이스 아티팩트을 호스트 바이너리에 내장하고 타입화된 실행 메서드을 생성합니다.`load_kernel_module`그리고 `cuda_launch!` `cuda_launch!`안전하지 않아서 포장해야 합니다`unsafe { }`.

:::{seealso}
[CUDA 프로그래밍 가이드 -- 실행 구성] ](https://docs.nvidia.com/cuda/cuda-programming-guide/#execution-configuration)에 대한 권위있는 참조에 대한 `<<<grid, block, smem, stream>>>` 의미.
:::

## 발사 수명 주기는

모든 커널 런칭은 같은 순서를 따르고 있습니다:

1. **CUDA 컨텍스트를 초기화합니다** - GPU 장치에 연결합니다.
2. **기기 모듈을 로드합니다** -- 보통 내장된 유물 팩에서요.
3. ** 커널 함수를 찾아보세요 ** - PTX 진입점 이름으로
4. **그리드를 구성하세요** - 블록 차원, 그리드 차원 공유 메모리
5. **발사** - 커널을 스트림에 줄지어 놓습니다.
6. ** 동기화** -- 결과 (명명 또는 암시) 를 기다립니다.

```{figure} images/launch-lifecycle.svg
:align: center
:width: 100%

커널 런칭 라이프 사이클. 호스트는 컨텍스트를 초기화하고 장치 모듈을 로드하고 그리드를 구성하고 타입이 지정된 방법을 통해 런칭합니다. GPU 스케줄러는 SM에 블록을 전송합니다.
```

실제, `#[cuda_module]`는 생성된 Rust API의 2~5 단계를 처리합니다. 일반적으로 컨텍스트 생성, `kernels::load`, 그리고 타입 지정 메서드 호출과 상호 작용합니다.

## `#[cuda_module]` -- 타입 지정 논증

커널을 인라인으로 `#[cuda_module]`타입 로더를 생성하는 모듈과 1개 에 한 가지 방법`#[kernel]`생성된 서명에서는 커널 논리를 확인하지만,`LaunchConfig`커널의 인덱싱 형태나 리소스 요구사항을 증명하지 않습니다. 따라서 원시 시작 방법은 안전하지 않습니다

```text
raw LaunchConfig   -> unsafe launch
PreparedLaunch<K>  -> safe launch of exactly K
```

여기서 "동시"는 스트림과 대기열을 즉시 제공한다는 것을 의미합니다. GPU 실행은 동기화 될 때까지 호스트를 겹칠 수 있습니다.

```rust
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
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

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();

    let a = DeviceBuffer::from_host(&stream, &[1.0f32; 1024]).unwrap();
    let b = DeviceBuffer::from_host(&stream, &[2.0f32; 1024]).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    // SAFETY: this is a 1D launch and all three buffers contain 1024 elements.
    unsafe {
        module.vecadd(&stream, LaunchConfig::for_num_elems(1024), &a, &b, &mut c)
    }
    .expect("Kernel launch failed");

    let result = c.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 3.0);
}
```

### 분야별 분포

                   | 설명|
|:-----------------------|:------------------------------------|
|`#[cuda_module]`       | 로더와 실행 메서드을 생성합니다|
|`kernels::load(&ctx)`  | 내장된 유물 팩을 로드합니다|
|`module.vecadd(...)`   | 타입 검사 문장; 원형 구성은 `unsafe`가 필요합니다|
|`LaunchConfig`         | 그리드/블록 크기와 smem|

### 논증 매핑

생성된 메소드는 커널 매개 변수를 호스트 값으로 매핑합니다:

|커널 매개 변수   | 호스트 논증          | GPU ABI|
|:-------------------|:-----------------------|:-----------------|
|`&[T]`             | `&DeviceBuffer<T>`     | 지표 + 길이가|
|`&mut [T]`         | `&mut DeviceBuffer<T>` | 지표 + 길이가|
|`DisjointSlice<T>` | `&mut DeviceBuffer<T>` | 지표 + 길이가|
|스칼라/크로 포인터 | 같은 값             | 직접 가치|

### 반환 값

원형 타입 지정 실행 메서드 반환`Result<(), DriverError>``Ok`case는 커널이 성공적으로 **열열되었다는 것을 의미합니다** - 완료된 것은 아닙니다. 실행 시간 오류 (예를 들어, 범위를 벗어난 함정) 를 확인하려면 스트림 또는 컨텍스트를 나중에 동기화합니다.

## 안전 준비 발사

커널의 기하학은 정확성의 일부일 때 선언합니다:

```rust
use cuda_core::LaunchConfig1D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};

#[cuda_module]
mod contracted {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

// SAFETY: this package embeds the artifact compiled from this `contracted` module.
let module = unsafe { contracted::load(&ctx)? };
let config = LaunchConfig1D::new(4, 256, 0);
let prepared = module.prepare_vecadd(config)?;
module.vecadd(&stream, &prepared, &a, &b, &mut c)?;
```

계약 모듈을 로딩하는 것은 안전하지 않습니다. 호출자는 선택한 유물이 이 모듈에서 컴파일되었는지 확인해야 하기 때문입니다. 일치하는 패키지 또는 커널 이름만으로 ABI와 런칭 요구 사항이 일치한다는 것을 증명하지는 않습니다. 올바른 유물이 로딩되면 준비는 선언된 런칭 계약을 확인합니다.

`prepare_vecadd`정확한 블록 형상, 장치 제한, 동적 공유 메모리, 컨텍스트 및 클러스터/협력 요구 사항을 검사합니다. `LaunchConfig1D`Y/Z의 활성 차원을 표현할 수 없고,`PreparedLaunch<vecadd>`다른 커널과 함께 사용할 수 없습니다. 준비는 실패 할 수 있습니다. 성공하면 브랜드 발사가 안전하게 재사용 될 수 있습니다

클러스터와 협동적 속성은 함께 선언될 수 있다 (`#[cluster_launch(...)]` + `#[cooperative_launch]`). 준비는 클러스트 형상을 `cuOccupancyMaxActiveClusters`로 증명하고, 그 다음 전체 그리드가 동시 클러터 용량에 맞게 되어 협동적인 `grid::sync()`가 비주민 블록에 매달릴 수 없도록 요구한다. 과대 크기의 그리드는 부적절한 런칭을 제출하기보다는 `LaunchContractError::CooperativeGridTooLarge` (또는 관련 클러스 레지던시 오류) 로 준비에 실패한다.

계약된 커널의 경우, 원시 탈출 래치는 `vecadd_unchecked`로 지정되어 안전하지 않습니다. 계약되지 않은 커널은 unsafe 원시 실행 메서드만을 노실행킵니다.

## 컴파일 타임 정책 구성

커널 정책은 컴파일 타임 선택의 이름집합이다. 그것은 타일 형상, 동작 원자, 실행 경계, 루프 풀링 요인 또는 커널 라이브러리가 노출하고자 하는 다른 구성을 설명할 수 있다.

정책은 실행 시간 값보다는  타입으로 표현된다. 일반 커널은 호스트가 사용하는 각 구체적인 정책 유형에 대해 한 번 단형화된다:

```text
transform::<SmallTilePolicy> -> transform_TID_<A>
transform::<WideTilePolicy>  -> transform_TID_<B>
```

각 전문화는 자체 PTX 입점과 자체 접힌 상수를 가지고 있습니다. 정책은 커널 인수로 전달되지 않으며 생성 된 커널에는 정책을 선택하는 실행 시간 가시가 포함되어 있지 않습니다.

### 도메인별 정책을 정의합니다

`cuda_device::config::Policy`는 의도적으로 안정적인 식별자만을 포함합니다. 커널 라이브러리는 해당 도메인에 의미있는 연관 유형과 상수로 확장합니다:

```rust
use cuda_device::config::{
    Atom, AtomKind, AtomSpec, Block, Global, Policy, PolicyId, RowMajor, Shape1, Thread, Tile,
    TileSpec,
};

enum XorRotate {}
impl AtomKind for XorRotate {}

trait VectorPolicy: Policy {
    type BlockTile: TileSpec<
        Layout = RowMajor,
        MemorySpace = Global,
        Scope = Block,
    >;

    type ElementAtom: AtomSpec<
        Kind = XorRotate,
        Scope = Thread,
    >;

    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const ITEMS_PER_THREAD: u32;
    const UNROLL: u32;
    const TAG: u32;
}
```

`Tile`, `Atom`, `Shape`, 레이아웃, 메모리 공간, 스코프 타입은 메타데이터이다. 이들은 메모리를 할당하지 않으며, 포인터 액세스를 제공하지 않으며 , 명령어를 발산하지 않으며 또는 스레드를 동기화하지 않는다. 도메인 특성과 커널 구현은 이러한 설명이 무엇을 의미하는지, 어떤 조합이 유효하는지 결정한다.

### 구체적인 정책을 정의합니다

구체적인 정책은 둘 다 구현합니다.`Policy`그리고 도메인별 정책 특성:

```rust
enum SmallTilePolicy {}

impl Policy for SmallTilePolicy {
    const ID: PolicyId =
        PolicyId::new(0x706f_6c69_6379_5f63, 1);
}

impl VectorPolicy for SmallTilePolicy {
    type BlockTile =
        Tile<Shape1<1024>, RowMajor, Global, Block>;

    type ElementAtom =
        Atom<XorRotate, Shape1<1>, Thread>;

    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
    const ITEMS_PER_THREAD: u32 = 16;
    const UNROLL: u32 = 2;
    const TAG: u32 = 0x1357_9bdf;
}
```

두 번째 정책은 커널을 중복하지 않고 다른 타일, 런칭bound 및 롤롤 인수를 선택할 수 있습니다:

```rust
enum WideTilePolicy {}

impl Policy for WideTilePolicy {
    const ID: PolicyId =
        PolicyId::new(0x706f_6c69_6379_5f63, 2);
}

impl VectorPolicy for WideTilePolicy {
    type BlockTile =
        Tile<Shape1<4096>, RowMajor, Global, Block>;

    type ElementAtom =
        Atom<XorRotate, Shape1<1>, Thread>;

    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
    const ITEMS_PER_THREAD: u32 = 16;
    const UNROLL: u32 = 4;
    const TAG: u32 = 0x2468_ace0;
}
```

`PolicyId`library-facing identity입니다. 이 이름 공간과 값은 정책 작성자가 제공하며 정책의 생성 동작이 변하지 않는 동안 안정적으로 유지되어야 합니다. 이 이름은 Rust에서 파생되지 않습니다.`TypeId`, 컴파일러 망글링, 타입 이름, 또는 컴파일에 의해 생성된 해시.

cuda-oxide는 글로벌 정책 ID 레지스트리를 유지하지 않습니다. 정책 라이브러리는 이름 공간을 선택하고 중복된 ID를 방지하는 데 책임이 있습니다.

### 커널에서 정책 상수를 사용

정책 관련 상수는 지원되는 컴파일 타임 속성으로 사용할 수 있습니다:

```rust
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub unsafe fn transform<P: VectorPolicy>(
        input: *const u32,
        output: *mut u32,
        count: u32,
    ) {
        let base =
            thread::index_1d().get()
                * P::ITEMS_PER_THREAD as usize;

        let mut lane = 0;

        #[unroll(P::UNROLL)]
        while lane < count {
            let index = base + lane as usize;

            // SAFETY: the caller provides readable and writable storage
            // for every active thread and keeps count within the tile.
            let value =
                unsafe { input.add(index).read_volatile() } ^ P::TAG;

            unsafe {
                output.add(index).write_volatile(value);
            }

            lane += 1;
        }
    }
}
```

rustc는 monomorphization 후에 관련 상수를 해결합니다. cuda-oxide는 따라서 각 전문에 대한 구체적인 값을 봅니다:

```text
SmallTilePolicy:
  launch_bounds = (64, 2)
  unroll         = 2

WideTilePolicy:
  launch_bounds = (256, 1)
  unroll         = 4
```

`MAX_THREADS`이 전문화에서 허용되는 최대 블록 크기를 지정합니다. 모든 발사에서 정확히 그 수의 스레드를 사용하도록 강요하지 않습니다.

`MIN_BLOCKS`PTX로 표시된 컴파일러 점유 지표입니다`.minnctapersm`그것은 그리드 또는 블록 차원이 아니며 발사 기하학으로 강제되지 않습니다.

`UNROLL`부분적인 분산 인수입니다.`UNROLL = 4`변환된 루프 몸당 4개의 루프 반복을 그룹화한다. 루프를 4개의 반복으로 제한하지 않는다. 런타임 경계와 나머지 반복은 보존된다.

유효하지 않거나 평가할 수 없는 정책 표현식은 기본적으로 값을 사용하지 않고 컴파일 오류를 발생시킵니다.

### 정책별 발사 준비

일반 커널은 정책 매개 변수가 구체적인 커널 전문성을 선택하는 일반 호스트 메소드를 생성합니다:

```rust
use cuda_core::LaunchConfig1D;

let small_launch =
    module.prepare_transform::<SmallTilePolicy>(
        LaunchConfig1D::new(1, 64, 0),
    )?;

let wide_launch =
    module.prepare_transform::<WideTilePolicy>(
        LaunchConfig1D::new(1, 256, 0),
    )?;
```

생성된 발사 계약은 각 전문 분야에 대해 독립적으로 평가됩니다:

* `SmallTilePolicy`는 64개의 스레드보다 큰 블록을 거부합니다.
* `WideTilePolicy`는 256개의 스레드보다 큰 블록을 거부합니다.

준비된 실행는 선택한 커널 전문에 대한 브랜드로 남아 있습니다. 한 정책에 대한 `PreparedLaunch`는 다른 정책의 런칭 메소드로 전달할 수 없습니다.

이 정책은 컴파일 타임 코드 생성 및 런칭 검증을 변경하지만, 원시 포인터 메모리 안전성을 증명하지 않습니다. 원시 포인트들을 수용하는 커널은 호출할 수 없습니다:

```rust
// SAFETY: the pointers cover the complete policy tile and remain valid
// until the stream completes the launch.
unsafe {
    module.transform::<SmallTilePolicy>(
        &stream,
        &small_launch,
        input,
        output,
        count,
    )?;
}
```

### 현재 제한 사항

정책 표현식은 현재 먼저 지원됩니다:

* `#[launch_bounds(...)]`
* 부분회로 `#[unroll(...)]`

일반 상수 표현식은 Rust의 `generic_const_exprs` 기능을 필요로 합니다:

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

현재 다음 값은 정책에서 파생된 값이 아닌 문자 그대로 유지됩니다:

* `launch_contract` 필드
* 클러스터 크기
* 동적 공유 메모리 크기

정책에서 파생된 최대 발사 제한은 여전히 호스트 발사 계약과 결합될 수 있다. 계약과 정책 최대는 각 단형화된 전문에 대해 별도로 확인된다.

정책에서 파생된 시작-결속 최대값에 참조되는 추가 이름의 상수는 모듈 범위에서 표시되어야 합니다. 왜냐하면 호스트 시작 계약이 커널 함수 옆에 생성되기 때문입니다.

### 전체 예제

`policy_config`]](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/policy_config) 예제에서 하나의 일반 커널에 대한 두 가지 정책을 정의하고, 그 정책이 생성되는 것을 확인합니다:

* 호스트가 주소로 할 수 있는 PTX 항목
* 별도의 `.maxntid` 및 `.minnctapersm` 지침
* 폴드된 정책별 상수
* 원료 LLVM IR의 다른 부분 롤링 인수
* 정책별 준비 발사 검증

GPU 없이 생성된 유물을 만들고 검사합니다:

```bash
cargo oxide build policy_config

cargo run --release \
  --manifest-path cuda-oxide/crates/rustc-codegen-cuda/examples/policy_config/Cargo.toml \
  -- --verify-ptx
```

컴파일러에서 생성된 전문화 이름이 빌드 프로파일마다 다를 수 있기 때문에 검증자를 위해 `--release`를 사용하십시오. `PolicyId`는 명시적이고 프로파일에 독립된 정책 정체성으로 남아 있습니다.

## `cuda_launch!` -- unsafe 낮은 수준의 발사

`cuda_launch!`아래의 명백하고 unsafe 탈출 입니다.`#[cuda_module]`. 그것의 틈새는 실행 시간에 이름에 의해 로드 모듈 (손으로 선택하는 사이드카 PTX / 큐빈 / LTOIR 유물), 어떤 컴파일 시간 커널 서명도 확인하는 데 존재합니다.

매크로는 논증 목록을 확인할 수 없기 때문에 모든 사용은 `unsafe { }` 블록 안에 있어야 합니다. 호출자는 논증 수, 순서 및 유형이 커널의 실제 서명과 일치하고 포인터 논증은 장치에 액세스 할 수 있다고 약속합니다. 불일치는 정의되지 않은 행동입니다. 드라이버는 args 배열의 끝을 지나 읽거나 장치가 쓰레기를 참조합니다.

```rust
use cuda_host::{cuda_launch, load_kernel_module};

let module = load_kernel_module(&ctx, "vecadd").unwrap();

// SAFETY: args match vecadd, buffers are live, and the config is 1D with
// bounds guarded by c.get_mut(idx).
unsafe {
    cuda_launch! {
        kernel: vecadd,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(1024),
        args: [slice(a), slice(b), slice_mut(c)]
    }
}
.expect("Kernel launch failed");
```

에 포장`args`생성된 호스트 패킷과 동일한 호스트 패켓을 생성합니다`#[cuda_module]`방법: `slice(...)`그리고 `slice_mut(...)`을 누르세요`(ptr, len)`쌍, 스칼라 논증들은 값을 직접 밀어내고, 닫기 또는 부가치 구조는 단일 부가치를 밀어냅니다 (핵 경계선에서는 하나의 로 수신합니다)`.param`, 각 필드별 평평한 매개 변수로 표시되지 않습니다).

## 유물 정책

`#[cuda_module]`이 정책은 목표 선택 기능이 아닌 실행 표면 기능입니다. 컴파일러가 호스트 바이너리에 배치한 임베디드 페이로드를 로드합니다. PTX 대 LTOIR, cubin 대 fatbin 또는 단일 아치 대 멀티 아치 패키징과 같은 결정은 컴파일에 라이브하고 유물 / 런타임 로더 레이어에서 라이브됩니다. 이 정책을 분리하여 생성된 러스트 실행 메소드가 페이로드 형식이 진화함에 따라 안정적으로 유지됩니다.

PTX 및 큐빈 페이로드가 직접 로드됩니다. NVVM IR는 Blackwell 전 GPU가 LLVM 7 타입 포인터를 사용하기 때문에 목표를 기록합니다. Blackwell 및 최신 GPU는 불투명한 포인터를 사용합니다. 로더는 libNVVM을 호출 할 때 기록된 목표를 사용합니다.

NVVM IR와 LTOIR는 원래의 목표에 대해 컴파일합니다.`sm_86`, 로더는 대신 PTX를 생성하고 CUDA 드라이버가 JIT을 블랙웰에 사용할 수 있습니다. 이것은 과 같은 후속 목표물에 지원되지 않습니다.`sm_90a`, 또는 오래된 GPU에서 최신 GPU 유물을 실행하기 위해. 드라이버는 또한 선택한 툴킷에서 생성된 PTX 버전을 지원해야합니다. CUDA 오류 222는 툴키트가 드라이버의 PTX JIT에 너무 새다는 것을 의미합니다. 호환 가능한 툴 키트를 선택하거나 드라이버를 업그레이드하십시오.{ref}`installation guide <installation-toolkit-driver-compatibility>`왜 이것이 일반적인 LLVM에서 PTX 경로와 다를 수 있는지 설명합니다.

## `LaunchConfig`

`LaunchConfig`는 그리드 형상을 지정합니다:

```rust
use cuda_core::simt::LaunchConfig;

let config = LaunchConfig {
    grid_dim: (num_blocks, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
};
```

|필드              | 타입              | 설명|
|:-------------------|:------------------|:--------------------------------|
|`grid_dim`         | `(u32, u32, u32)` | x, y, z에 있는 블록의 수|
|`block_dim`        | `(u32, u32, u32)` | x, y, z에서 블록당 스레드|
|`shared_mem_bytes` | `u32`             | 블록당 동적 공유 메모리|

### `for_num_elems`도움자

1차원 데이터 병렬 커널의 일반적인 패턴은 요소당 하나의 스레드입니다:

```rust
let config = LaunchConfig::for_num_elems(N as u32);
```

이것은 블록당 256개의 스레드를 사용하며 천장 분할을 통해 그리드 크기를 계산합니다:`grid_x = (N + 255) / 256`그것은 편리한 1D 형태이지만 여전히 원료 데이터입니다. 준비만 커널에 구성을 묶습니다.

### 2D 및 3D 구성

행렬 연산에 2차원 블록 및 격자 크기를 사용:

```rust
let config = LaunchConfig {
    grid_dim: ((cols + 15) / 16, (rows + 15) / 16, 1),
    block_dim: (16, 16, 1),
    shared_mem_bytes: 0,
};
```

커널 내부에서, 행 및 열 인덱스를 계산하기 위해 `threadIdx_x()` / `blockIdx_x()`와 `_y()`를 결합합니다.

### 블록 크기를 선택

블록 크기는 가장 중요한 단 하나의 조정 매개 변수입니다 ( 참조){ref}`Execution Model <execution-choosing-block-size>`자세한 내용은 다음 장에서 확인합니다.

- **256**는 대부분의 커널에 대한 안전 기본값입니다.
- 2의 힘 (128, 256, 512) 은 경전 경계에 맞춰져 있습니다.
- `#[launch_bounds]`를 사용하여 컴파일러에 원하는 블록 크기를 알려줍니다.

## 타입 지정 비동시 시작

과 함께`cuda-host`아시크 기능이 활성화된 상태입니다.`#[cuda_module]`또한 빌려와 소유의 비동시 메소드를 생성합니다. 이 게으른 반환 `DeviceOperation`실행 시에는 스트림이 지정되지 않습니다. 스케줄링 정책은 동작이 실행될 때 하나를 선택합니다:

```rust
use cuda_async::simt::device_context::init_device_contexts;
use cuda_async::simt::device_operation::DeviceOperation;

init_device_contexts(0, 1)?;
let module = kernels::load_async(0)?;

// SAFETY: this is 1D, buffers contain 1024 elements, and module/scheduler share a context.
let op = unsafe {
    module.vecadd_async(
        LaunchConfig::for_num_elems(1024),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )
}?;

// Execute and wait
op.sync()?;
```

운영을 생성하거나 저장해야 할 경우 소유 된 양식을 사용하십시오`'static`미래:

```rust
use std::future::IntoFuture;

// SAFETY: this is 1D, owned buffers contain 1024 elements, and contexts match.
let op = unsafe {
    module.vecadd_async_owned(
        LaunchConfig::for_num_elems(1024),
        a_dev,
        b_dev,
        c_dev,
    )
}?;

let (a_dev, b_dev, c_dev) = tokio::spawn(op.into_future()).await??;
```

### 아시크 버퍼 수명

비동기 발사는 게으름 때문에 포인터 수명이 중요합니다.

```text
raw pointer shape:
  build op from CUdeviceptr
  drop buffer
  run op later  -> stale pointer

borrowed typed shape:
  build op from &DeviceBuffer
  Rust keeps the buffer borrowed until op completes

owned typed shape:
  move DeviceBox into op
  spawned task owns the allocation until completion
```

계약된 커널의 경우, 두 아시크 형식 모두 `&PreparedLaunch<K>`를 받아 안전합니다. `cuda_launch_async!`는 낮은 수준의 unsafe 마이그레이션 API로 남아 있습니다. 그 호출은 `unsafe` 블록 내에서 이루어져야 합니다. 러블러는 할당이 게으른 동작보다 오래 지속되는 것을 증명할 수 있을 때만 RAW 포인터 아시클 실행가 정확합니다.

### `.sync()` 대 `.await`

|방법    | 무슨 일이죠?|
|:----------|:----------------------------------------------------------------------------------|
|`.sync()` | 기본 스케줄링 정책을 실행, 완료 될 때까지 현재 스레드를 차단|
|`.await`  | 현재 아시크 작업을 실행하고 생성합니다 (토쿄 런타임이 필요합니다)|

## GPU 작업을 작성하는 것

`DeviceOperation`기능적 구성을 지원합니다.`and_then`그리고 동시에 독립적인 일을 합니다.`zip!`

```rust
use cuda_async::zip;

let forward_pass = layer1_op
    .and_then(|output1| layer2_op(output1))
    .and_then(|output2| layer3_op(output2));

// Run two independent operations concurrently
let combined = zip!(branch_a, branch_b);
let (result_a, result_b) = combined.sync()?;
```

체인 내의 각 연산은 실행될 때만 스트림에 스케줄된다. `and_then` 조합기는 한 연산의 출력을 다음 연산에 입력으로 전달하여 게으른 계산 그래프를 형성한다.

:::{seealso}
[아시크 GPU 프로그래밍]](../async-programming/the-device-operation-model.md) 섹션에서는 `DeviceOperation`, 스케줄링 정책 및 스트림 관리를 깊이 다루고 있습니다.
:::

## 클러스터 발사

스레드 블록 클러스터 (호퍼 및 최신 버전) 는 블록이 ** 분산 공유 메모리** (DSMEM) 를 통해 공유 메모리를 넘어 협력 할 수 있습니다. 클러터를 실행하려면 커널에 `#[cluster_launch]`를 추가하고 시작에 `cluster_dim`을 포함하십시오:

```rust
use cuda_device::{DisjointSlice, cluster, cluster_launch, kernel};

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(mut out: DisjointSlice<u32>) {
    let rank = cluster::block_rank();
    // Blocks 0-3 can communicate via DSMEM
}
```

호스트에서 발사에는 을 사용합니다.`launch_kernel_ex`(확장된 발사 API) 클러스터 크기와 함께.`cuda_launch!`을 통해 지원합니다.`cluster_dim`필드:

```rust
// SAFETY: args/config match cluster_kernel, including its 4-block cluster;
// out_dev stays live through synchronization.
unsafe {
    cuda_launch! {
        kernel: cluster_kernel,
        stream: stream,
        module: module,
        config: config,
        cluster_dim: (4, 1, 1),
        args: [slice_mut(out_dev)]
    }
}
.expect("Cluster launch failed");
```

:::{tip}
클러스터 시작은 ** 호퍼 (sm_90) ** 또는 최신 버전이 필요합니다. 최대 클러스트 크기는 일반적으로 16 블록입니다.`cargo oxide build --arch sm_90`홉퍼를 겨냥해
:::

## 협동 발사

그리드 전체 장벽 (`cuda_device::grid::sync()` 또는 `this_grid().sync()`) 은 그리드의 모든 블록이 동시에 장치에 거주 할 때만 작동합니다. ** 협동 시작 **은 드라이버에게 정확히 보장하도록 요청합니다. 그렇지 않으면 아직 계획되지 않은 블록은 장벽과 커널 정결점에 도달 할 수 없습니다.

타이핑을 하고`#[cuda_module]`경로, 으로 커널을 표시`#[cooperative_launch]`. 생성된 모든 시작 방법 ( 동기화, 비동기, 그리고 소유-비동기) 는 을 통해 제출됩니다.`cuLaunchKernelEx``CU_LAUNCH_ATTRIBUTE_COOPERATIVE`속성 집합:

```rust
use cuda_device::{DisjointSlice, cooperative_launch, grid, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[cooperative_launch]
    pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {
        // ... per-block work ...
        grid::sync();
        // ... grid-wide post-barrier work ...
    }
}

let module = kernels::load(&ctx)?;
// SAFETY: config satisfies the kernel's indexing, residency, and output bounds.
unsafe { module.grid_sync_kernel(&stream, config, &mut out_dev) }?;
```

다른 것 처럼`#[cluster_launch]`, 이 속성은 PTX에서 아무것도 변경하지 않습니다. 호스트가 실행을 제출하는 방식만 변경합니다. 두 속성은 하나의 커널에서 결합 될 수 있습니다. 이 경우 두 실행 속성은 동일한 에 들어가 있습니다.`cuLaunchKernelEx`전화해

레거시 (콜러 unsafe) `cuda_launch!` 매크로는 그 `cooperative: true` 필드를 통해 동일한 동작을 제공합니다.

:::{tip}
전체 그리드는 단 하나의 파동으로 장치에 맞아야 하며, 그렇지 않으면 운전자는 `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`로 발사하는 것을 거부합니다. 의심할 경우 그리드를 `cuOccupancyMaxActiveBlocksPerMultiprocessor` (SM × SM 수당 블록) 에서 크십시오.
:::

## 일반적인 발사 오류

|오류                                  | 유력한 원인                                           | 고쳐|
|:---------------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------|
|`CUDA_ERROR_INVALID_VALUE`             | 그리드 또는 블록 크기는 0이거나 제한을 초과합니다     | 체크`LaunchConfig`값; 최대 블록은 1024 스레드|
|`CUDA_ERROR_NOT_FOUND`                 | PTX 입구 이름과 일치하지 않습니다                     | 확인해`#[kernel]`이름은 로딩된 모듈과 일치합니다|
|`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`   | 너무 많은 공유 메모리 또는 블록당 너무 많은 레지스터 | 줄여`shared_mem_bytes`또는 블록 크기; 사용`#[launch_bounds]`|
|`CUDA_ERROR_ILLEGAL_INSTRUCTION`       | 커널이 함정에 빠졌어요 (공황, 주장 실패, OOB)         | `cargo oxide debug` 또는 `gpu_printf!`로 디버그|
|`CUDA_ERROR_NO_BINARY_FOR_GPU`         | PTX가 잘못된 아키텍처를 위해 컴파일된                    | GPU와 일치하는 `--arch`로 재구성|
|`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222) | 드라이버는 모듈에서 PTX 버전을 컴파일할 수 없습니다 | 호환된 `CUDA_TOOLKIT_PATH` 또는 드라이버를 업그레이드|
|`CUDA_ERROR_NOT_INITIALIZED`           | `libcuda`가 없어졌거나 운전자의 CUDA 전공이 도구장치보다 오래되었습니다 | R580 + 드라이버를 설치; 오류 텍스트는 로더가 시도한 파일의 이름을|
|`CUDA_ERROR_NOT_FOUND` (발사되지 않은 호출에서) | 로딩된 `libcuda`에서 드라이버 기호가 없어지고 있다     | 툴킷의 CUDA 버전으로 드라이버를 업그레이드|

:::{seealso}
[실점 처리 및 디버깅] ](error-handling-and-debugging.md) 장은 커널 오류를 진단하고 수정하는 방법을 자세히 설명합니다.
:::

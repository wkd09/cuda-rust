# 메모리 및 데이터 이동

GPU는 호스트와 분리된 자체 메모리를 가지고 있다. 장치로 데이터를 가져와서 올바른 종류의 메모리를 선택하는 것은 모든 CUDA 프로그램에 필수적이다. 이 장은 호스트/기기 전송에서 공유 메모리와 커널 ABI에 이르기까지 cuda-oxide의 메모리 추상화를 다루고 있다.

:::{seealso}
[CUDA 프로그래밍 가이드 -- 장치 메모리]](https://docs.nvidia.com/cuda/cuda-programming-guide/#device-memory-accesses) CUDA 메모리 계층 및 액세스 패턴에 대한 권위있는 참조를 위해.
:::

## CUDA 메모리 계층

NVIDIA GPU는 여러 계층의 메모리를 노출하며, 각각 용량, 대기 시간 및 범위가 다릅니다.

|메모리력                     | 적용 범위       | 전형적인 크기                 | 지연시간          | -오산화 API|
|:---------------------------|:------------|:-----------------------------|:-----------------|:-------------------------------------|
|**등록물**              | 실당  | ~255 × 32비트                | 0회~         | 지역 변수|
|**공유 메모리**          | 블록당   | 48-228 KB (아치에 의존)  | ~5회~        | `SharedArray`, `DynamicSharedArray`|
|** L1 캐시 **               | SM당      | 공유와 결합된         | 하드웨어 관리 | 자동으로|
|**상속 메모리**        | 모든 스레드 | 64 KB, 단독으로 읽기용   | 방송 캐시 | `ConstantMemory`를 통해 `#[constant]`|
|** L2 캐시 **               | 칩 전체   | 최대 50 MB (호퍼)         | ~30회~       | 자동으로|
|**전역 메모리 (DRAM) **   | 모든 스레드 | 16~80 GB (HBM)              | ~400회~      | `DeviceBuffer`, `DeviceBox`|

가이드 원칙: 자주 액세스되는 데이터를 더 빠르고 가까운 메모리로 이동하십시오. 레지스터는 가장 빠르지만 스레드 당; 공유 메모리는 빠르고 블록 전체에 표시됩니다; 전역 메모리는 크고 느립니다.

```{figure} images/memory-hierarchy.svg
:align: center
:width: 100%

CUDA 메모리 계층은 가장 빠른 (레지스터, 스레드당) 에서 가장 큰 (글로벌 DRAM, 모든 스레드와 함께) 으로부터. 각 계층이 지연 용량을 교환합니다. 오른쪽 패널은 각 레벨의 범위와 cuda-oxide API를 보여줍니다.
```

## 맥락과 흐름

메모리 API에 뛰어들기 전에, 모든 코드 예제에서 나타나는 두 개의 호스트 측면 개념은 소개가 필요합니다: ** 컨텍스트 ** 및 ** 스트림 **.

**CUDA 컨텍스트** (`CudaContext`) 는 호스트 스레드를 특정 GPU에 묶습니다. 그것은 그 장치의 모든 리소스를 소유하고 있습니다. 모듈, 스트림, 할당. 일반적으로 프로그램 시작에서 하나를 만듭니다.

```rust
use cuda_core::CudaContext;

let ctx = CudaContext::new(0).unwrap();   // bind to GPU 0
```

**CUDA 스트림** (`CudaStream`) 은 GPU 동작의 순서대로 이루어진 줄을 의미한다. ** 같은 ** 스트림에 순서로 있는 동작은 FIFO 순서에서 실행된다. 각각은 전작 동작의 모든 부작용을 볼 수 있다. ** 다른 ** 스트림을 하는 동작은 서로 순서 가맹 없이 겹쳐서 동시에 실행될 수 있다

```rust
let stream = ctx.default_stream();     // the implicit, always-available stream
let work_stream = stream.fork()?;      // a new stream, joined to the parent's current position
```

모든 메모리 전송 및 커널 런칭에는 스트림이 필요합니다. 단일 스트림 프로그램 (이 책에서 대부분의 예제를 포함합니다) 에 대해서는 ** 기본 스트림 **이 필요합니다 - 모든 것이 연속적이고 구조적으로 올바릅니다. 멀티 스트림 파이프 라인은 컴퓨팅과 데이터 이동 사이의 중복을 잠금하지 않지만 이벤트 또는 `join`를 통해 명시적인 동기화를 필요로합니다:

```text
Same stream:       [kernel_A] → [memcpy_B] → [kernel_C]     (automatic ordering)
Different streams: [kernel_A on stream 1] | [memcpy_B on stream 2]  (concurrent, needs events)
```

```{figure} images/cuda-streams.svg
:align: center
:width: 100%

위: FIFO에서 자동으로 명령되는 단일 스트림 실행. 아래: A와 B 스트림이 동시에 실행되는 멀티 스트림 수행, 커널_쓰기 및 커널 _읽기 사이의 데이터 의존성을 설정하는 이벤트.
```

:::{seealso}
[Kernels Launching]](launching-kernels.md) 장은 런칭 매크로에서 스트림 사용에 대해 설명하고, [Async GPU 프로그래밍]](../async-programming/the-device-operation-model.md) 섹션은 스트림을 자동으로 관리하는 `DeviceOperation`를 다루고 있다.
:::

## `DeviceBuffer` -- 호스트/장치 전송

`DeviceBuffer<T>``cuda_core`기기 메모리를 할당하고 호스트와 GPU 사이에서 데이터를 이동하는 주된 방법입니다:

```rust
use cuda_core::{CudaContext, DeviceBuffer};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();

// Host → device: copy a host slice to GPU memory
let a_dev = DeviceBuffer::from_host(&stream, &host_data).unwrap();

// Allocate zeroed device memory
let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

// Device → host: read results back
let results = c_dev.to_host_vec(&stream).unwrap();
```

### 주요 방법

|방법                            | 방향     | 설명|
|:--------------------------------- |:--------------|:-------------------------------|
|`from_host(&stream, &[T])`        | 호스트 → 장치 | 할당 + 차단 복사본|
|`zeroed(&stream, len)`            | --            | 할당 + 0 채우기|
|`to_host_vec(&stream)`            | 장치 → 호스트 | 비동기 복사 + 반환 `Vec<T>`|
|`copy_to_host(&stream, &mut [T])` | 장치 → 호스트 | 기존 슬라이스에 복사|
|`cu_deviceptr()`                  | --            | 생화질`CUdeviceptr`FFI에 대해|

### 소유권 및 매각

`DeviceBuffer`을 통해 분배를 자유롭게 합니다.`cuMemFree`이것은 차단 드라이버 호출입니다. 비행 중 Kernel가 메모리에 닿지 않도록 장치 전체를 내부적으로 동기화합니다.

- ** 커널이 실행 중인 동안 떨어지는 것은 **
GPU는 완전히 비활성 상태이고, 메모리를 자유롭게 해 주세요.
- ** 동기화 후 떨어지는 ** (예: `to_host_vec` 또는
`stream.synchronize()`이 장치가 이미 사용되지 않기 때문에 추가 비용이 없습니다.

단일 스트림 워크로드에 대해서는 괜찮습니다. 모든 것이 FIFO 순서로 실행됩니다. 결과를 읽을 때 모든 커널이 완료되고 무료는 즉각적입니다. 비용이 멀티 스트림 시나리오에서 눈에 띄게됩니다. 컴퓨터와 메모리 작업을 중복시키고 싶은 경우; 하나의 스트림에서 동기화 된 무료는 다른 모든 스트림의 작업을 중단 할 수 있습니다.

## `DeviceBox` -- 비동기 친화적인 장치 메모리

이 작업은 `DeviceBox<T>`에서 `cuda_async`에서 동기화 자유 문제를 해결합니다. 드롭에서, ** 전용 딜로커 스트림 **에 대한 `cuMemFreeAsync`를 통해 메모리를 자유롭게합니다. 이것은 스트림 순서 작업입니다. - 무료는 딜로카터 스트림에 대기하고 그 스트림의 모든 전작 작업이 완료된 후에만 실행됩니다. 중요하게는 ** 장치를 동기화하지 않습니다:

```rust
use cuda_async::simt::device_box::DeviceBox;
use cuda_async::simt::device_context::init_device_contexts;

init_device_contexts(0, 1)?;  // Initialize device context map (default device 0)

// DeviceBox wraps a device pointer; freed asynchronously on drop
let dev_ptr: DeviceBox<f32> = /* allocated by DeviceOperation chain */;
// When dev_ptr is dropped, cuMemFreeAsync is called on the deallocator stream.
// Other streams continue running without stalling.
```

### 둘 중 하나를 선택하는 것

                     | `DeviceBuffer`                      | `DeviceBox`|
|:--------------------|:------------------------------------|:---------------------------------------|
|*조금자리*           | `cuda_core`                         | `cuda_async`|
|*수송이 무료*    | `cuMemFree` (시크 -- 스탠드 장치) | `cuMemFreeAsync` (아시크 -- 스탠드 없이)|
|****와 함께 사용        | 타이핑 동기화 발사                 | 타입의 비동시 발사|
|호스트의 읽기   | `to_host_vec()`                     |
|**최고의 **        | 일류, 차단 작업 부하   | 다중 흐름, 파이프라인 작업 부하|

:::{tip}
멀티 스트림 파이프 라인에서의 지연 민감성 해제에는 `DeviceBox`를 선호한다. 간단한 단일 스트림 예제에서는 `DeviceBuffer`가 더 간단하고 동기 자유는 사실상 비용이 없다.
:::

(메모리-어거먼트-스칼라리제이션) =

## 인수 스칼라리제이션

당신이 `&[f32]`를 사용하는 커널을 작성할 때 호스트와 디바이스가 메모리에  슬라이스를 표현하는 방법에 동의하지 않습니다. 구조 레이아웃은 호스트의 x86 ABI와 NVPTX ABI 사이에 다를 수 있습니다. cuda-oxide는 이것을 커널 경계에서 ** 스칼라리제** 집계 타입으로 해결합니다. 두쪽 모두 동일하게 해석하는 원시 값으로 분해합니다.

|커널 파라미터 타입          | 호스트가 실제로 전달하는 내용|
|:-------------------------------|:-------------------------------|
|`&[T]`                         | `ptr: *const T` + `len: u64`|
|`DisjointSlice<T>`             | `ptr: *mut T` + `len: u64`|
|`T` (스칼라)                   | `T`직접|
|`#[grid_constant] value: &T`    | 한 부가치 `T`; 장치가 발사 저장소를 빌려; 호스트 발사는 안전하지 않습니다|
|구조`{ a: u32, b: f32 }`    | 1개의 부가 값 (전체 구조)|
|폐쇄 (N 포획)      | 1개의 부가 값 (전체 구조)|
|제로 사이즈 타입               | 완전히 벗겨졌어요|

그래서 쓰게 된 거야`#[cuda_module]`방법들 받아들여`&DeviceBuffer<T>``&[T]`그리고 `&mut DeviceBuffer<T>`작성 가능한 슬라이스 같은 매개 변수들을 위해. 생성된 메소드는 포인터와 길이를 추출합니다. 커널 내부에서 컴파일러는 스칼라 매개변수로부터 슬라이스의 구조를 재구성합니다. 그래서 커널 코드는 정상 을 보죠.`&[T]`종류.

```{figure} images/scalarization-abi.svg
:align: center
:width: 100%

논증 스칼라리제이션: 호스트는 ABI 경계를 통해 (ptr, len) 쌍으로  슬라이스를 전달합니다. 장치 커널은 평면 스칼러 매개 변수를 수신하고 컴파일러는 커널 내부의 원래  타입을 재구성합니다.
```

:::{tip}
스칼라리화는 정상적인 커널 코드에서 완전히 보이지 않습니다. 서명에 `&[f32]`를 쓰고 일반 슬라이스로 사용합니다. 생성 된 런칭 메소드와 컴파일러는 다른 모든 것을 처리합니다.
:::

## `DisjointSlice` -- 안전 평행 문장

CUDA C++에서 병렬 출력 표준 패턴은 모든 스레드가 인덱스하는 원형 `__global__` 포인터입니다. 이것은 본질적으로 안전하지 않습니다. 두 개의 스레드를 동일한 위치에 작성하는 것을 막을 수있는 것은 없습니다.

을 공급합니다`DisjointSlice<T, IndexSpace>`안전한 대안으로. 그것은 변동성 슬라이스를 을 통해만 쓸 수 있습니다.`ThreadIndex`누구의 `IndexSpace`준비된 발사 기하학과 결합하면 각 스레드가 고유한 요소에 액세스 할 수 있습니다.

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn double(input: &[f32], mut out: DisjointSlice<f32>) {
    if let Some((out_elem, idx)) = out.get_mut_indexed() {
        *out_elem = input[idx.get()] * 2.0;
    }
}
```

- 이 `get_mut_indexed()`는 단회 통화 형태입니다.
그리고 으로 풀립니다.`&mut T`한 발로 `None`두 개의 네트워크 밖의 스레드를 모두 덮습니다 (예를 들어, `col >= ROW_STRIDE`2D) 및 슬라이스 외 지수.
- 두 단계의 명시적인 형태 `let idx = thread::index_1d(); out.get_mut(idx)`
또한 여러 조각에 걸쳐 평행 수학적 인덱스가 필요할 때 사용할 수 있습니다.
- 여러 스레드가 의도적으로 쓰이는 절감과 같은 패턴에
같은 위치로, `get_unchecked_mut` (불안) 은 탈출 뚜을 제공합니다.

### 왜?`ThreadIndex`안전해지죠

의 열쇠`DisjointSlice`안전이`ThreadIndex<'kernel, IndexSpace>`공개적인 구성자가 없는 불투명한 증인이죠. 이를 얻는 유일한 방법은 하드웨어 내장 변수 () 에서 값을 도출하는 신뢰할 수 있는 인덱스 함수 (index functions) 를 통해서입니다.`threadIdx`, `blockIdx`, `blockDim`):

```rust
let idx = thread::index_1d();          // ThreadIndex<'_, Index1D> -- ok
let bad = ThreadIndex::new(42);        // does not exist -- private constructor
```

이것은 CUDA의 스레드 인덱스가 하드웨어에서 제공하는 ** 균일한 값이기 때문에 작동합니다: 블록의 모든 스레드는 GPU의 워프 스케줄러에서 고유한 `threadIdx`를 수신합니다. 1D 그리드 런칭 (x 차원만 사용되는 경우), `blockIdx.x * blockDim.x + threadIdx.x`에서 파생된 글로벌 인덱스는 전체 그리드에서 고유합니다.

마지막 진술은 조건이 있다: `index_1d()`는 Y와 Z를 무시한다. 2차원 격자는 모든 Y 좌표에 대해 동일한 X-파생 지수를 반복한다.

```text
PreparedLaunch(domain=1) -> Y/Z inactive -> safe DisjointSlice writes
raw LaunchConfig         -> caller proves Y/Z inactive -> unsafe launch
```

증인은 장치 측면 지표 공간을 증명하고 준비된 발사는 호스트 측면 기하학을 증명합니다. 둘 다 완전히 안전한 발사에 필요합니다.

증인은 또한 `!Send + !Sync + !Copy + !Clone`, 그리고 그것의 `'kernel` 수명은 매크로 주입하는 스택 로컬 스코프에서 빌려온 것이다. 그래서 스레드는 이웃이 나중에 가져갈 수 있도록 공유 메모리에 그것의 `ThreadIndex`를 주차할 수 없으며, 증인은 커널 본체를 생존할 수 없다. 이것을 `IndexSpace` 매개 변수 (`Index1D`, `Index2D<S>`, `Runtime2DIndex`) 와 결합하면 타입 시스템 또한 컴파일 시에 부합되지 않은 2D 걸음을 거부한다. 데이터 레이스 위험은 타입 오류가 된다.

## 공유 메모리

**공유 메모리**는 블록 내의 모든 스레드에 가시적인 빠른 칩 내 메모리이다. 그것은 속도와 범위 모두에서 레지스터 (스레드당) 와 전역 메모리 (모든 스레드) 사이에 앉아 블록 내에서 스레드와 데이터 재사용을 위한 주요 도구이다.

### 정적 공유 메모리 -- `SharedArray`

컴파일 시 크기가 알려져 있을 때 `SharedArray<T, N>`를 사용한다:

```rust
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

#[kernel]
pub fn smem_example(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let tid = thread::threadIdx_x() as usize;

    // Load from global to shared
    unsafe { TILE[tid] = input[idx.get()]; }
    thread::sync_threads();

    // Read neighbor from shared (much faster than global)
    let neighbor = if tid > 0 {
        unsafe { TILE[tid - 1] }
    } else {
        0.0
    };
    thread::sync_threads();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = unsafe { TILE[tid] } + neighbor;
    }
}
```

각 `static mut SharedArray`별도의 지도 `.shared`PTX로 할당됩니다.`sync_threads()`모든 스레드가 읽기 전에 작성 완료된 것을 보장합니다.

### 동적 공유 메모리 -- `DynamicSharedArray`

크기가 실행 시간 매개 변수에 따라 달라지면 `DynamicSharedArray<T>`를 사용 하 고 `LaunchConfig::shared_mem_bytes`을 통해 할당 크기를 지정 합니다:

```rust
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread};

#[kernel]
pub fn dynamic_smem_example(input: &[f32], mut out: DisjointSlice<f32>) {
    let smem = DynamicSharedArray::<f32>::get();
    let tid = thread::threadIdx_x() as usize;

    unsafe { *smem.add(tid) = input[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... use smem ...
}
```

호스트에서, 시작에서 크기를 설정:

```rust
let config = LaunchConfig {
    grid_dim: (num_blocks, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
};
```

여러 개의 동적 배열은 `DynamicSharedArray::offset(byte_offset)`를 사용하여 동일한 할당을 공유할 수 있습니다.

### 정렬

|타입                          | 기본 정렬  | 참고문서|
|:------------------------------|:-------------------|:-----------------------------|
|`SharedArray<T, N>`           | `align_of::<T>()`  | 표준 정렬|
|`SharedArray<T, N, 128>`      | 128 바이트          | TMA 작업에 필요한|
|`DynamicSharedArray<T>`       | 16 바이트           | nvcc-compatible 기본 설정|
|`DynamicSharedArray<T, 128>`  | 128 바이트          | TMA에 필요한 사항|

### 일반적인 함정

- **미수된 `sync_threads()`:** 공유 메모리 입력 사이의 장벽이 없는
그리고 읽으면, 스레드는 오래된 데이터나 초기화되지 않은 데이터를 읽을 수 있습니다.
- **SM 한계를 초과:** 너무 많은 공유 메모리를 요청하는 원인
`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`건축의 한계를 확인해 보세요.

:::{seealso}
[CUDA 프로그래밍 가이드 -- 공유 메모리] ](https://docs.nvidia.com/cuda/cuda-programming-guide/#shared-memory) 은행 갈등, 방송 및 최적 접근 패턴에 대한 자세한 사항
:::

## 지속적인 메모리력 --`#[constant]`

상시 메모리는 호스트가 시작과 그리드의 모든 스레드를 읽는 사이에 작고 읽기 전용 창입니다. **방송 **를 위해 구축된 캐시가 서비스됩니다. 워크의 모든 경로가 동일한 주소를 원하면 하나의 거래이며 다른 주소를 원하는 경우 순차적으로 서비스됩니다..

로 선언합니다`static` 안에`#[cuda_module]`

```rust
#[cuda_module]
mod kernels {
    use cuda_device::{ConstantMemory, DisjointSlice, constant, kernel, thread};

    #[constant]
    static SCALE: ConstantMemory<f32> = ConstantMemory::UNINIT;

    #[kernel]
    pub fn multiply(mut out: DisjointSlice<f32>) {
        let s = SCALE.get();
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            *e = (i as f32) * s;
        }
    }
}
```

매크로는 한 상수당 2개의 호스트 세터를 생성합니다.`set_`그리고 정적의 이름을 소문자로 적어 놓습니다.`SCALE`이 됩니다.`set_scale`그리고 `TABLE_CONST`이 됩니다.`set_table_const`일반 형태는 스트림 순서로 되어 있습니다.`set_<name>_blocking`형제는 대신 동시 한 장의 복사본을 만듭니다:

```rust
let module = kernels::load(&ctx)?;
module.set_scale(&stream, &3.0)?;
```

이 `ConstantMemory<T>`는 원시 정수와 플로이트, `*const T` / `*mut T`, 최대 8개의 요소에 이르는 튜플 및 그 배열에 대해 구현된 unsafe 마커인 `T: ConstantMemoryValue`를 수용한다. 이 `ConstantMemory::UNINIT`가 0으로 시작되기 때문에 모든 0 비트 패턴이 유효한 `T`이며, 이 때문에 참조와 `NonZero*` 유형이 배제되고, 사용자 지정 구조가 선택하기 전에 명시적인 `#[repr(C)]` 레이아웃이 필요한 이유이다.

### `get()` 대 `get_ref()`

이것은 스칼라를 제외한 다른 어떤 것에 사용하기 전에 알아야 할 유일한 것입니다.

`get()`부가치 복사본을 반환합니다. 값은 주소가 없습니다, 그래서 을 인덱싱`ConstantMemory<[f32; N]>`실행 시간 인덱스에 복사하면 전체 배열이 먼저 스레드의 로컬 디포로 쏟아져 나오게 됩니다. 각 스레드마다 하나의 저장소, 그리고 검색은 스레드에 대한 개인 메모리를 읽습니다.`get_ref()`저장소를 빌려서, 그래서 지수는 일정한 공간에 남아 있고, 읽기 단일 `ld.const`.

`constant_memory_table`예를 들어 두 가지 대안 모두 세 번째 옵션에 대해`const T: [f32; N]`A10G (sm_86) 에서 측정된 256개의 항목`f32`테이블, 8388608개의 스레드에 걸쳐 64개의 의존적 검색, 모든 변종은 비트 동일:

|지수        | `get()` | `get_ref()` | `const [f32; N]` (전 세계)|
|:-------------|--------:|------------:|--------------------------:|
|분산    | 16048 us |      7616 us |                 322주|
|워크 유니폼 | 16065us |   **258us** |                     322 us|

두 가지 결론.`get_ref`2 x/`get`이 경우, 이 경우 이 경우의 경우, 양변수와 양변형의 경우의 양변수가 62x이고, 양각형의 경우에는 62x입니다. 따라서 양변수를 초과하는 것을 선호합니다. 그리고 상속 메모리는 단순히 "더 빠른 메모리"가 아닙니다: 이 경우 23.7x * 평상시 배열 변수보다 뒤쪽에 있습니다. 왜냐하면 양변의 양각수와 작은 테이블의 양반수들이 결합하여 L1에 거주하고 있는 반면, 상속 캐시는 순서대로 다른 주소들을 서비스합니다.`#[constant]`지표가 -일률적으로 되면 평평한 지표로 도달합니다`const`array가 아닌 경우

랜치에 걸쳐 읽기 접혀지지 않습니다: `UnsafeCell` 내부는 컴파일러에 관해서는 저장소를 변경할 수 있도록 유지하므로 랜치를 사이에 있는 `set_<name>`는 눈에 띄어 있습니다. 랜치가 계속 반복되는 경우에도 랜치는 통합될 수 있습니다.

:::{seealso}
[CUDA 프로그래밍 가이드 -- `__constant__`(https://docs.nvidia.com/cuda/cuda-programming-guide/#constant) 이 지도에 하드웨어 제한 및 캐시 규칙을 위해.
:::

## 통일 메모리 및 HMM

기본 설정으로 GPU는 CPU에서 ** 별도의 주소 공간 **에서 작동합니다. GPU는 일반적인 호스트 포인터를 지칭할 수 없습니다. 주소는 단순히 GPU의 페이지 테이블에 아무 것도 매핑하지 않습니다. 따라서 전통적인 CUDA 워크플로우는 장치 메모리에 명시적 할당을 요구하고 명시적 복사본이 따라갑니다:

```text
┌──────────────────┐              ┌──────────────────┐
│   CPU Memory     │    PCIe /    │   GPU Memory     │
│   (host DRAM)    │◄────────────►│   (device HBM)   │
│                  │   NVLink     │                  │
└──────────────────┘    copy      └──────────────────┘
   Separate address spaces -- GPU cannot dereference host pointers
```

CUDA는 GPU가 첫 번째 액세스 시 페이지 오류 지연의 비용으로 투명하게 호스트 메모리에 액세스하도록 함으로써이 제한을 완화시키는 메커니즘을 제공합니다.

### 한눈에 메모리 액세스 모드

|모드                 | 어떤 GPU에 액세스할 수 있나요?   | 할당 필요    | 첫 번째 접근 비용            | 하드웨어 요구 사항|
|:---------------------|:----------------------|:-----------------------|:-----------------------------|:--------------------------|
|**명료 복사**    | 단장 메모리만    | `DeviceBuffer`         | 아무 것도 (예전에 복사된 데이터)   | 어떤 CUDA GPU|
|**기록 (지도) **  | 특정 호스트 버퍼 | `cudaHostAlloc`        | 높은 (~ 10-20 μs 접근) | 어떤 CUDA GPU|
|**통합 메모리 **   | 관리 할당자료   | `cudaMallocManaged`    | 중위 (페이지 마이그레이션)      | 케플러+ (sm_30+)|
|**HMM**              | *어떤* 호스트 메모리     | 아무 것도 없어요                   | 중위 (페이지 오류 + 검색)  | 리눅스에서 튜링+|

cuda-oxide는 주로 ** Explicit Copy ** (`DeviceBuffer`, `DeviceBox`) 를 대용량 데이터와 ** HMM **를 이동 폐쇄 캡처 및 작은 구성 데이터에 사용합니다.

### 통일 메모리력

통일 메모리 (Unified Memory) 는 CUDA의 관리 메모리 할당자 (`cudaMallocManaged`) 이다. 생성된 포인터는 CPU와 GPU 모두에서 유효하다. CUDA 런타임은 프로세서가 각 페이지를 "소유"하고 요구에 따라 마이그레이션하는 것을 추적합니다. GPU가 현재 호스트 DRAM에 있는 페이지에 액세스 할 때, 런타임을 통해 데이터를 읽기 전에 장치 메모리에 투명하게 복사합니다. 이 마이그레이션은 코드에 보이지 않지만 무료가 아닙니다. "잘못된" 쪽에서 첫 번째 액세스하면 페이지 오류가 발생하고 상호 연결을 통해 DMA 전송이 발생합니다. 동일한 페이지에 대한 후속 액세스에서는 GPU의 로컬 캐시를 쳤습니다.

cuda-oxide는 현재 `cudaMallocManaged`를 직접 포장하지 않습니다. 관리 메모리 워크플로우에서는 원형 결합을 통해 CUDA 드라이버 API를 사용해야합니다. 실제로는 `DeviceBuffer::from_host` (명확한 복사) 는 대부분의 사용 사례를 다루고 예측 가능한 성능을 제공합니다.

### HMM (이질적인 메모리 관리)

HMM는 유니피드 메모리 (Unified Memory) 의 요구 페이지링 모델을 **모든 시스템 메모리** - 힙 할당, `mmap` 영역, 심지어 스택 변수까지 확장하는 리눅스 커널 기능이다. HMM가 활성화되면 GPU는 특별한 CUDA 할당자가 없이도 유효한 호스트 포인터를 탈 참조할 수 있다:

```rust
let factor = 5i32;                           // ordinary stack variable
let scale = |x: i32| x * factor;            // captures &factor (non-move)
// SAFETY: args match the kernel's signature; &factor stays alive via HMM.
unsafe { cuda_launch! { kernel: scale, args: [...] } }  // GPU reads &factor via HMM
```

통일 메모리와 달리 HMM는 특별한 할당 API가 필요하지 않습니다. 포인터는 단순한 호스트 주소입니다. ATS (Address Translation Services) 가 Grace Hopper와 같은 하드웨어-동일성 플랫폼에서 사용할 때 HMM을 대체하고 캐시 라인 곡성에서 하드웨어 일관성을 제공합니다. HMM은 자동으로 비활성화됩니다.

### 페이지 오류에 무슨 일이 일어나는지

커널이 장치 메모리에 해당되지 않는 주소에서 로드되면 하드웨어와 드라이버가 협력하여 검색합니다:

1. SM는 가상 주소를 위해 글로벌 로드 (`ld.global`) 를 실행합니다.
2. **GPU MMU**는 TLB에서 주소를 검색합니다. 실수하면, 그것은 을 걸습니다
장치 페이지 테이블.
3. 페이지 테이블에 매핑이 없다면 GPU는 ** 페이지 오류 **를 일으킨다.
이 다른 을 계속할 수 있습니다.
4. **CUDA 드라이버 오류 처리기**는 페이지의 소스를 결정합니다:
   - *통합 메모리* - CUDA 실행시간은 관리 할당을 식별하고
이동을 시작하게 됩니다.
   - *HMM* -- 리눅스 커널의 HMM 레이어는 호스트 가상 주소를 해결합니다.
호스트 페이지를 핑하고, 이 페이지를 마이그레이션하거나 원격 지도를 만듭니다.
5. PCIe 또는 NVLink을 통해 **DMA 전송**은 호스트 DRAM에서 페이지를 복사합니다
장치 HBM. GPU 메모리 컨트롤러는 데이터를 기록하고 호스트 메모리 컨트롤러는 읽기 서비스를 제공합니다.
6. GPU 페이지 테이블이 업데이트되고 TLB가 다시 채워지고 ** warp
이 페이지는 이제 로컬이고 L2에 캐시되어 있습니다. 후속 접속은 수백 회에 불과합니다.

단계 5의 지연 시간은 상호 연결에 달려 있습니다:

|연결           | 대역폭  | 오류 지연시간 | 참고문서|
|:-----------------------|:-----------|:--------------|:---------------------------------------|
|PCIe 4.0 x16           | ~25 GB/s   | ~10~20 μs    | 대부분의 데스크톱/워크스테이션 GPU|
|PCIe 5.0 x16           | ~50 GB/s   | ~5-15 μs     | 아다 로블레이스 + 더 새로운 플랫폼|
|NVLink 4.0             | ~900 GB/s  | ~1~5 μs      | 데이터 센터 GPU (H100, B100)|
|그레이스 홉퍼 C2C       | ~900 GB/s  | <1 μs         | 하드웨어 코레언스 -- ATS를 사용하지만 HMM를 사용하지 않습니다|

결함이 페이지의 곡성 (4 KB 또는 2 MB) 에서 작동하기 때문에 단일 결함이 많은 스레드를 만족시킬 수 있습니다. 워프 레벨 합성 또한 도움이됩니다: 연속적인 4 바이트 요소를 읽는 32 개의 스레드는 최대 하나 또는 두 페이지에 닿지 않습니다. PCIe 시스템에서는 단일 결함 비용이 대략 작은 `cudaMemcpy`와 같습니다. 요구 페이지의 장점은 실제로 만지는 페이지에만 지불한다는 것입니다.

### 쿠다-옥시드가 HMM을 사용하는 방법

cuda-oxide는 HMM를 두 가지 방법으로 활용합니다.

1. **동동이 없는 닫는 캡처. * * 비-`move` 닫는가 에 전달될 때
커널에서 캡처된 변수는 호스트 스택에 남아 있고 GPU는 HMM 포인터를 통해 액세스합니다. 이것은 커널이 한 번 또는 드물게 읽는 데이터를 복사하는 것을 피합니다.

2. *다이나믹 레이아웃의 구조 ABI.** Cuda-oxide는 Rust의 실제 구조와 일치합니다
배열 ( 포함)`#[repr(Rust)]`필드 재질서) 를 장치 쪽으로 표시하여 HMM에 액세스하는 호스트 구조는`#[repr(C)]`또는 수동 레이아웃 사양. 컴파일러는 `rustc` 뚜렷한 패딩을 가진 LLVM 구조형과 일치하는 필드 오프셋 및 빌드.

### HMM 시스템 요구 사항

|요구 사항      | 최소|
|:-----------------|:-------------------------------------|
|GPU 아키텍처 | 튜링 (컴퓨터 능력 7.5+)|
|리눅스 커널     | 6.1.24+, 6.2.11+, 또는 6.3+|
|CUDA 드라이버      | 535+ 오픈 커널 모듈|

HMM가 시스템에 활성화되었는지 확인합니다:

```bash
nvidia-smi -q | grep Addressing
# Addressing Mode : HMM  ← HMM is enabled
```

### HMM 대 명시적 복사본을 사용할 때

|시나리오                                    | 권장 접근법|
|:--------------------------------------------|:-------------------------------------------|
|많은 스레드에 의해 처리되는 큰 배열      | `DeviceBuffer::from_host` (명확한 복사본)|
|소규모 읽기 전용 구성 데이터          | HMM (패스 포인터, GPU 페이지 오류를 허용)|
|CPU와 GPU가 반복적으로 공유하는 데이터 | 두 번 버퍼링된 명시적 복사|
|프로토타입 제작/빠른 실험             | HMM (단순 -- 복사본이 필요없다)|

:::{tip}
HMM은 성능 전략이 아니라 편리성입니다. 대역폭에 민감한 커널의 경우 페이지 오류 오버헤드를 피하고 전체 메모리 버스 너비를 사용하기 때문에 장치 메모리에 명시적인 복사 항상 더 빠르다.
:::

:::{seealso}
[CUDA 프로그래밍 가이드 -- 통일 메모리] ](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) 및 [NVIDIA 블로그 -- HMM로 GPU 개발을 단순화] ](https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/) 페이지 마이그레이션, 사전 검색 및 시스템 요구 사항에 대한 자세한 내용은.
:::

# 공유 메모리와 동기화

CUDA 스레드 블록의 모든 스레드는 **공유 메모리**라는 작고 빠른 임시 저장 공간을 사용할 수 있습니다. 공유 메모리는 SM 실행 유닛 옆의 칩 내부에 있으며 전역 메모리보다 약 100배, L1 캐시보다 약 10배 빠릅니다. 단점은 용량입니다. 아키텍처에 따라 SM당 보통 48~228KB이며, 해당 SM에서 실행되는 모든 블록이 나누어 사용합니다.

cuda-oxide는 `SharedArray`와 `DynamicSharedArray`를 통해 공유 메모리를 제공합니다. 두 타입은 Rust 배열처럼 사용할 수 있고 PTX 주소 공간 3으로 컴파일됩니다. 이 장에서는 두 타입의 사용법, 동기화 시점, 단순한 커널을 빠르게 만드는 타일링 패턴을 설명합니다.

:::{seealso}
하드웨어 뱅크 구조, 브로드캐스트 규칙, 아키텍처별 용량은 [CUDA 프로그래밍 가이드 — 공유 메모리](https://docs.nvidia.com/cuda/cuda-programming-guide/#shared-memory)를 참조하세요.
:::

---

## 공유 메모리가 중요한 이유

[비동기 MLP 파이프라인](../projects/async-mlp-pipeline.md) 프로젝트의 단순 GEMM을 살펴보겠습니다. 각 스레드는 전역 메모리에서 A의 행 전체와 B의 열 전체를 읽어 출력 원소 하나를 계산합니다. 64×64 행렬에서는 스레드마다 전역 메모리 로드가 128회 발생하고, 블록 안의 다른 스레드도 같은 원소를 여러 번 읽습니다. 하드웨어는 각 값을 DRAM에서 가져옵니다. 운이 좋으면 L2 캐시에 있는 값을 가져올 수도 있습니다.

공유 메모리를 사용하면 비용 구조가 달라집니다. 스레드 블록이 협력해 A와 B의 **타일**을 공유 메모리에 적재하고 동기화한 뒤, 모든 스레드가 해당 타일에서 읽습니다. 각 전역 메모리 로드는 그 원소를 재사용하는 스레드 전체에 분산됩니다. 16×16 타일을 사용하면 전역 메모리 트래픽을 16분의 1로 줄일 수 있습니다.

```{figure} images/shared-memory-tiling.svg
:align: center
:width: 100%

타일 기반 계산 패턴입니다. 스레드들이 협력해 A와 B의 타일을 전역 메모리에서 `SharedArray`로 불러오고 동기화합니다. 빠른 칩 내부 메모리에서 부분 곱을 계산한 뒤 다시 동기화하고, K 차원의 다음 타일에 대해 반복합니다.
```

---

## SharedArray — 정적 공유 메모리

`SharedArray<T, N, ALIGN>`은 컴파일 시점에 공유 메모리에 할당되는 고정 크기 배열입니다. 커널 안에서 `static mut`으로 선언합니다.

```rust
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

const TILE: usize = 16;

#[kernel]
pub fn tiled_sgemm(
    m: u32, n: u32, k: u32,
    a: &[f32], b: &[f32],
    mut c: DisjointSlice<f32, Runtime2DIndex>,
) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

    let n_sz = n as usize;
    let row = thread::index_2d_row();
    let col = thread::index_2d_col();
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;

    let mut sum = 0.0f32;
    let mut t = 0u32;

    while t < k / TILE as u32 {
        let tile_offset = t as usize * TILE;

        // 1단계: 협력 로드
        unsafe {
            TILE_A[ty * TILE + tx] = a[row * k as usize + tile_offset + tx];
            TILE_B[ty * TILE + tx] = b[(tile_offset + ty) * n_sz + col];
        }

        // 스레드가 읽기 전에 모든 스레드의 로드가 끝나야 합니다
        thread::sync_threads();

        // 2단계: 공유 메모리에서 계산
        let mut i = 0usize;
        while i < TILE {
            unsafe {
                sum += TILE_A[ty * TILE + i] * TILE_B[i * TILE + tx];
            }
            i += 1;
        }

        // 다음 반복에서 타일을 덮어쓰기 전에 동기화해야 합니다
        thread::sync_threads();

        t += 1;
    }

    // The row width comes from `c`, bound on the host to this same `n`.
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        if let Some(c_elem) = c.get_mut(c_idx) {
            *c_elem = sum;
        }
    }
}
```

### 선언 규칙

`SharedArray`는 커널 함수 본문 안에서 `static mut`으로 선언해야 합니다. 그래야 컴파일러가 PTX 공유 주소 공간(`.shared`)에 할당합니다. `UNINIT` 상수는 초기화를 생략하므로 스레드가 값을 쓰기 전까지 내용은 정의되지 않습니다.

| 매개변수 | 의미                                                                           |
| :-------- | :-------------------------------------------------------------------------------- |
| `T`       | 원소 타입(`f32`, `u32` 등)                                                 |
| `N`       | 원소 개수(컴파일 시 고정)                                        |
| `ALIGN`   | 바이트 정렬: 0(자연 정렬) 또는 2의 거듭제곱. TMA 대상에는 128 사용       |

:::{tip}
`SharedArray`가 `!Sync`인 것은 의도된 설계입니다. 내부에서 `UnsafeCell`을 감싸므로 컴파일러가 스레드 간 불변성을 가정하지 않습니다. 공유 메모리는 블록 내 모든 스레드가 수정할 수 있으므로 올바른 동작이며, 동기화는 프로그래머가 책임져야 합니다.
:::

### 정렬 규칙

`ALIGN`은 `0`(자연 정렬) 또는 0이 아닌 2의 거듭제곱이어야 합니다. 다른 값이면 사용 위치를 표시하는 컴파일 오류가 발생합니다.

저장 공간의 정렬은 `align_of::<T>()`보다 작게 선언되지 않습니다. 요청값이 더 작으면 `#[repr(align(N))]`이 구조체 정렬을 높이는 것처럼 조용히 상향되고, 더 큰 요청값은 그대로 적용됩니다. 따라서 `SharedArray<u64, 2, 1>`은 8바이트 정렬로 선언되며, 자연 정렬을 지정한 `#[repr(C, align(16))]` 원소는 16바이트 정렬로 선언됩니다. 정렬을 높이지 않으면 1바이트 정렬의 `u64` 배열이 바이트 배열 바로 뒤에 배치되어 `misaligned address` 오류가 날 수 있습니다.

자연 정렬을 사용하는 집계형, 중첩 배열, `f16` 원소는 PTX에 자연 정렬을 선언합니다(`[f32; N]`은 `.align 4`, `f16`은 `.align 2`). 이전 릴리스는 이들에 일괄적으로 `.align 8`을 출력했습니다.

```{figure} images/shared-memory-alignment.svg
:align: center
:width: 100%

공유 메모리 저장 공간의 정렬입니다. 검증기는 사용 위치에서 0이 아닌 2의 거듭제곱이 아닌 `ALIGN` 값을 거부합니다. 하위 변환 단계는 정적 배열과 동적 풀을 요청 정렬과 원소 ABI 정렬 중 큰 값으로 선언합니다. `SharedArray<u64, 2, 1>`은 `.align 8`로 선언되고 자연 정렬의 `#[repr(C, align(16))]` 원소는 `.align 16`으로 선언됩니다. 이전 릴리스는 각각 `.align 1`, `.align 8`로 선언해 커널에서 `misaligned address` 오류가 날 수 있었습니다.
```

### API

```rust
impl<T, const N: usize, const ALIGN: usize> SharedArray<T, N, ALIGN> {
    pub const UNINIT: Self;
    pub const fn len() -> usize;
    pub fn as_ptr(&self) -> *const T;
    pub fn as_mut_ptr(&mut self) -> *mut T;
    pub unsafe fn as_raw_mut_ptr(shared: *mut Self) -> *mut T;
}

// Indexing (unsafe access via static mut)
impl Index<usize> for SharedArray<T, N, ALIGN> { ... }
impl IndexMut<usize> for SharedArray<T, N, ALIGN> { ... }
```

디버그 빌드에서는 인덱스 범위를 검사합니다. 릴리스 빌드와 GPU에서는 범위 검사를 생략합니다. 범위를 벗어나 인덱싱하면 Rust의 다른 `static mut` 접근과 마찬가지로 정의되지 않은 동작이 발생합니다.

---

## DynamicSharedArray — 런타임 크기 할당

타일 크기를 컴파일 시점에 알 수 없거나 여러 논리 배열이 하나의 공유 메모리 풀을 나누어 써야 할 때가 있습니다. `DynamicSharedArray`는 동적 공유 메모리 영역에서 할당하며, 영역 크기는 실행 시 `LaunchConfig::shared_mem_bytes`로 지정합니다.

```rust
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread};

#[kernel]
pub fn reduce_dynamic(input: &[f32], n: u32, mut output: DisjointSlice<f32>) {
    let tid = thread::threadIdx_x() as usize;

    // 동적 공유 메모리 영역의 포인터를 가져옵니다
    let smem: *mut f32 = DynamicSharedArray::<f32>::get();

    unsafe {
        // 전역 메모리에서 불러옵니다
        let idx = thread::index_1d();
        *smem.add(tid) = if idx.get() < n as usize {
            input[idx.get()]
        } else {
            0.0
        };
    }

    thread::sync_threads();

    // 트리 축약
    let mut stride = thread::blockDim_x() as usize / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                *smem.add(tid) += *smem.add(tid + stride);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let block_idx = thread::blockIdx_x() as usize;
        unsafe {
            *output.get_unchecked_mut(block_idx) = *smem;
        }
    }
}
```

다음과 같이 실행 설정을 지정합니다.

```rust
let config = LaunchConfig {
    grid_dim: ((n + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
};
```

### 동적 공유 메모리 분할

하나의 동적 풀에서 여러 배열을 사용하려면 `offset()`을 이용하세요.

```rust
let pool_a: *mut f32 = DynamicSharedArray::<f32>::get();
let pool_b: *mut f32 = DynamicSharedArray::<f32>::offset(
    256 * std::mem::size_of::<f32>()
);
```

`offset`은 풀 시작점으로부터의 **바이트 오프셋**을 받습니다. 전체 사용량이 `shared_mem_bytes`를 넘지 않도록 하세요. 런타임 검사는 없습니다.

### 정렬

`DynamicSharedArray`의 기본 정렬은 16바이트입니다. TMA 작업(Hopper 이상)에는 필요한 128바이트 정렬을 얻도록 `DynamicSharedArray<f32, 128>`을 사용하세요. 정렬은 PTX `.align` 지시문으로 인코딩됩니다.

`ALIGN`은 0이 아닌 2의 거듭제곱이어야 하며, `0`이면 사용 위치를 표시하는 컴파일 오류가 발생합니다. 커널에는 동적 풀이 하나만 있습니다. 컴파일러는 커널에서 사용한 `get`, `get_raw`, `offset`의 타입별 요구 정렬(각각 `ALIGN`과 `align_of::<T>()` 중 큰 값)과 실행 계약의 정렬을 합쳐 가장 큰 값으로 풀을 선언합니다. `DynamicSharedArray::<u64, 1>::get()`만 사용해도 풀은 8바이트 정렬로 선언됩니다. 선언은 풀의 기본 주소만 정렬합니다. `offset(byte_offset)` 결과를 `T`에 맞게 정렬하는 것은 호출자 책임입니다.

### 타입 지정 공유 포인터: `SharedPtr`

`DynamicSharedArray::get()`과 `SharedArray::as_mut_ptr()`는 `*mut T`를 반환합니다. 디바이스 코드에서 이는 전역, 로컬, 공유 메모리(클러스터의 다른 블록 공유 메모리 포함)를 가리킬 수 있는 *일반* 포인터입니다. 이런 포인터를 헬퍼 함수에 전달하면 컴파일러는 해당 포인터가 이 블록의 공유 메모리를 가리키는지 더 이상 알 수 없습니다.

`SharedPtr<T>`는 호출 블록의 공유 메모리(`.shared::cta`)를 가리키는 원시 포인터이며, 타입에 그 사실을 담습니다. 헬퍼 매개변수와 반환값, 구조체 필드, 제네릭 코드 어디서나 `addrspace(3)` 포인터로 하위 변환됩니다. 따라서 이 포인터를 사용하는 명령은 일반 주소를 변환하지 않고 공유 메모리 형식을 사용합니다.

```rust
use cuda_device::{DynamicSharedArray, SharedArray, SharedPtr, kernel, thread};

#[inline(never)]
unsafe fn fill(tile: SharedPtr<f32>, i: usize, value: f32) {
    // Still a shared-memory pointer here, although `fill` is not inlined.
    unsafe { tile.add(i).as_ptr().write(value) };
}

static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

#[kernel]
pub fn fill_tile() {
    // SAFETY: TILE is a `static mut SharedArray` of this kernel.
    let tile: SharedPtr<f32> = unsafe { SharedArray::shared_ptr(&raw mut TILE) };
    // The same type for dynamic shared memory, 1024 bytes into the pool.
    let pool: SharedPtr<u8> = DynamicSharedArray::<u8, 128>::shared_ptr_at(1024);
    // SAFETY: the launch supplies more than 1024 dynamic shared bytes, and
    // each thread writes its own slot of TILE.
    unsafe {
        fill(tile, thread::threadIdx_x() as usize, 1.0);
        pool.as_ptr().write(0);
    }
}
```

| 연산                              | 결과                                              |
|:---------------------------------------|:----------------------------------------------------|
| `unsafe SharedArray::shared_ptr(&raw mut S)` | 이 커널의 `static mut`인 `S` 첫 원소를 가리키는 `SharedPtr<T>` |
| `DynamicSharedArray::<T>::shared_ptr()` / `shared_ptr_at(bytes)` | `get()` / `offset(bytes)`와 동일 |
| `unsafe SharedPtr::from_raw(p)`        | 일반 포인터 `p`가 이 블록의 공유 메모리를 가리킨다고 단언 |
| `add`, `offset`, `byte_add`, `cast`    | 타입을 유지하는 포인터 연산              |
| `as_ptr()`                             | 로드, 저장, 원시 포인터 API용 일반 `*mut T` |
| `addr_u32()`                           | `_shared_u32` 형식에 쓰는 32비트 `.shared::cta` 주소 |

`SharedPtr`는 단독으로든 구조체나 배열 안에 있든 커널 매개변수가 될 수 없습니다. 호스트에는 전달할 공유 메모리 주소가 없으며 컴파일러가 해당 시그니처를 거부합니다. `cluster::map_shared_rank`에서 얻은 포인터는 다른 블록의 공유 메모리를 가리킬 수 있으므로 `SharedPtr`가 아닙니다.

집계 타입 안에서는 일반적인 공유 포인터 제한이 적용됩니다. 유니온 저장 규칙에 따라 유니온이나 `MaybeUninit<SharedPtr<T>>`는 안전하게 거부됩니다. 이를 포함하는 `repr(packed)` 구조체에는 packed 보유자 제한이 적용됩니다. 두 제한 모두 [지원 기능](../appendix/supported-features.md)에 나와 있습니다. libNVVM 경로에서 공유 포인터는 4바이트이므로 `SharedPtr`와 더 좁은 필드(예: `{ p: SharedPtr<u8>, x: u32 }`)를 섞은 구조체는 아직 배열이나 공유 메모리에서 지원되지 않습니다. 이 제약은 `SharedPtr`보다 먼저 존재했으며 `*mut Barrier` 필드에도 동일하게 적용됩니다.

---

## 동기화: sync_threads()

`thread::sync_threads()`는 **블록 전체 배리어**입니다. PTX `bar.sync 0`으로 컴파일되며 다음 두 가지를 보장합니다.

1. 어떤 스레드도 진행하기 전에 **블록의 모든 스레드**가 배리어에 도달합니다.
2. 배리어 이후 **모든 스레드의 메모리 쓰기**가 다른 모든 스레드에 보입니다(공유 메모리의 메모리 펜스 역할).

로드 단계와 계산 단계 사이에 `sync_threads()`가 없으면 다른 스레드가 쓰기 전에 일부 스레드가 공유 메모리 위치를 읽을 수 있습니다. 하드웨어는 공유 메모리 저장에 대해 워프 내 순서를 보장하지 않습니다. 배리어가 없으면 같은 워프의 스레드끼리도 오래된 값을 볼 수 있습니다.

### 동기화 시점

규칙은 간단합니다. **다른 스레드가 쓴 값을 읽기 전에 동기화하세요.**

| 상황                                                      | 동기화 필요?                                                               |
| :--------------------------------------------------------------| :------------------------------------------------------------------------|
| 스레드 A가 `TILE[i]`에 쓰고 스레드 B가 읽음            | 예                                                                      |
| 스레드 A가 `TILE[i]`에 쓰고 같은 스레드가 읽음            | 아니요(같은 스레드)                                                         |
| 다음 반복을 위해 타일 덮어쓰기                 | 예(새 로드가 다른 스레드가 읽는 데이터를 덮기 전에) |
| `DisjointSlice`에서 읽기(각 스레드가 자기 인덱스를 읽음) | 아니요                                                                       |

앞의 타일 GEMM에는 반복마다 동기화 지점이 두 곳 있습니다. 하나는 로드 후(계산 전), 다른 하나는 계산 후(다음 로드가 타일을 덮어쓰기 전)입니다. 둘 중 하나라도 빠지면 데이터 경합이 발생합니다.

:::{tip}
흔한 실수는 모든 스레드가 실행하지 않는 조건 분기 안에 `sync_threads()`를 두는 것입니다. 블록의 모든 스레드가 같은 `sync_threads()` 호출에 도달해야 하며, 그렇지 않으면 커널이 교착 상태에 빠집니다. 분기 제어 흐름이 필요하다면 배리어를 분기 밖으로 옮기도록 구조를 바꾸세요.
:::

(compiler-ordering-around-the-barrier)=

### 배리어 주변의 컴파일러 명령 순서

위 하드웨어 보장은 스레드 간 가시성을 다룹니다. 별도로 컴파일러가 각 메모리 접근을 작성한 배리어의 앞뒤에 유지하는지도 확인해야 합니다. 두 백엔드 모두 컴파일러가 모든 메모리를 읽고 쓰는 작업으로 취급하는 배리어를 출력합니다.

- LLVM NVPTX 경로는 `llvm.nvvm.barrier.cta.sync.aligned.all`을 호출합니다. 이 호출은 `convergent`이며 `memory(...)` 속성은 없습니다.
- libNVVM 경로는 `~{memory}` 클로버가 있는 `asm sideeffect`로 `bar.sync 0;`을 출력합니다.

| 접근 형태                                              | `sync_threads()`의 해당 위치에 유지되는가 | 설명 |
| :------------------------------------------------------- | :----------------------------------- | :---- |
| `SharedArray` 인덱싱, `S[i]`                           | 예 | 주소 공간 3의 일반 로드 또는 저장입니다. |
| `SharedArray::as_raw_mut_ptr(&raw mut S)`                | 예 | 인덱싱과 동일한 명령을 사용합니다. `as_mut_ptr()`와 `as_ptr()`보다 이 메서드를 권장합니다. 두 메서드는 `S`를 빌리며, 2024 에디션에서는 `static_mut_refs` 린트가 `static mut` 빌림을 거부합니다. |
| `DynamicSharedArray::<T>::get()` 및 `offset()`          | 예 | 반환된 포인터를 통한 일반 로드 또는 저장입니다. |
| 옵션 없는 `ptx_asm!`                               | 예 | 메모리 속성 없이 `asm sideeffect` 호출로 처리됩니다. PTX는 배리어, 일반 로드·저장, 다른 부작용 어셈블리와의 순서를 유지합니다. 결과를 사용하지 않는 코드는 ptxas가 SASS에서 제거할 수 있습니다. |
| `clobber("memory")`가 있는 `ptx_asm!`                      | 예 | 제약 문자열에 `~{memory}`를 추가합니다. 현재 하위 변환에서는 위 행과 같은 순서를 유지합니다. 클로버가 없으면 순서는 `sideeffect` 플래그만으로 유지되지만, 이 플래그는 코드 조각이 메모리에 접근한다고 컴파일러에 알리지 않습니다. 주소를 통해 로드나 저장을 수행하는 모든 코드 조각에 추가하세요. |
| `options(register_only)`가 있는 `ptx_asm!`                 | 아니요 | 호출에 `sideeffect` 플래그가 없습니다. 컴파일러는 메모리 작업과 배리어를 넘어 코드를 이동할 수 있으며 결과가 사용되지 않으면 제거합니다. 레지스터만 사용하는 코드 조각에 쓰세요. |
| `cp_async_*` 복사, `cp_async_commit_group()`, `cp_async_wait_group(n)` | 예, 아래 설명 참조 | 실행, 커밋, 대기 순서는 유지됩니다. 배리어는 복사 완료를 기다리지 않습니다. |
| 다른 스레드가 쓰는 슬롯을 가리키며 배리어를 넘어 살아 있는 `&T` 또는 `&[T]` | 아니요 | 어느 백엔드보다 먼저 rustc가 이를 통한 읽기를 병합합니다. 아래 설명을 참조하세요. |
| 해당 슬롯을 가리키는 `&UnsafeCell<T>` | 예 | 각 배리어 뒤에 다시 읽는 일반 로드입니다. 사이에 다른 작업이 없는 인접한 두 읽기는 하나의 로드로 합쳐질 수 있습니다. |
| `BlockAtomic*` 타입에 대한 참조 | 예 | 각 `load()`는 메모리 클로버가 있는 부작용 어셈블리입니다. 컴파일러는 모든 로드를 유지합니다. |

인덱싱, `as_raw_mut_ptr`, `DynamicSharedArray`, 두 종류의 부작용 있는 `ptx_asm!` 행은 생산자·소비자 커널(저장, 배리어, 로드, 배리어, 저장, 배리어, 로드)로 두 백엔드에서 확인했습니다. PTX는 각각의 순서를 유지합니다. 반복 횟수가 커널 인자인 루프 안의 로드도 순서를 유지합니다. RTX 5060 Laptop(sm_120)에서 실행해 예상한 이웃 값을 읽었습니다. 결과가 사용되지 않는 `register_only` `ld.shared`는 PTX에 명령을 남기지 않지만 옵션이 없는 동일한 코드는 유지됩니다. `cp.async`와 참조 행은 아래에서 별도로 테스트합니다.

이 표는 컴파일러 명령 순서만 다룹니다. 배리어 위치는 여전히 “동기화 시점”의 규칙에 따라 결정해야 합니다.

#### 배리어 전후의 원시 포인터

각 스레드는 할당의 원시 주소에서 포인터를 만들며 배열 전체에 대한 참조는 생성하지 않습니다. 스레드는 배리어 전에 자기 슬롯에 쓰고, 배리어 뒤에 이웃 슬롯을 읽습니다.

```rust
static mut TILE: SharedArray<u32, 128> = SharedArray::UNINIT;

let tid = thread::threadIdx_x() as usize;
let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
unsafe { tile.add(tid).write(tid as u32 + 1) }; // this thread's slot
thread::sync_threads();
let next = unsafe { tile.add((tid + 1) % 128).read() }; // a neighbour's slot
```

이 코드 조각으로 만든 커널은 두 백엔드 모두에서 128개 스레드 전부 이웃 값을 읽었습니다.

#### 배리어 전에 `cp.async` 대기 필요

`bar.sync`는 대기 중인 `cp.async` 완료를 기다리지 않습니다. 각 스레드는 자기 복사를 기다린 다음 모든 스레드가 배리어에 도달해야 합니다.

```rust
unsafe {
    cp_async_ca_4(dst, src);
    cp_async_commit_group();
    cp_async_wait_group(0); // this thread's copies have landed
}
thread::sync_threads(); // other threads may now read `dst`
```

복사가 끝나기 전에 대상 주소를 읽으면 정의되지 않은 동작입니다. `cp_async_wait_group(0)`을 생략하고 배리어 뒤에 이웃 슬롯을 읽는 테스트 커널은 오래된 값을 반환할 수 있습니다. sm_120에서 일반 실행했을 때 128개 스레드 모두 오래된 값을 읽었고, compute-sanitizer에서는 읽기 전에 모든 복사가 완료되었습니다. 대기를 추가한 같은 커널은 128개 스레드 모두 복사된 값을 반환했습니다.

#### UnsafeCell을 통해서만 참조로 슬롯 공유

`T`에 내부 가변성이 없으면 공유 메모리를 가리키는 `&T`는 참조가 살아 있는 동안 대상이 바뀌지 않는다고 rustc에 알립니다. rustc는 백엔드 실행 전에 해당 참조를 통한 두 번의 읽기를 하나의 로드로 합칩니다. 테스트 커널에서 각 스레드가 `&u32`로 이웃 슬롯을 읽고, 이웃 스레드가 새 값을 쓰는 동안 두 번 배리어를 통과한 뒤 다시 읽었습니다. 두 백엔드 모두 128개 스레드에서 두 번째 읽기가 이전 값을 반환했습니다. 타일에 `from_raw_parts`로 만든 `&[T]`도 마찬가지입니다. 배리어를 가로지르는 두 인덱스 읽기가 하나의 로드로 컴파일됩니다. 다른 스레드가 대상을 쓰는 동안 이런 참조를 배리어 너머로 유지하면 정의되지 않은 동작입니다. `&mut T`도 마찬가지입니다. 다른 스레드의 쓰기가 독점 접근을 깨뜨리기 때문입니다(현재 읽기 병합 패스가 없더라도 그렇습니다).

`&UnsafeCell<T>`는 각 배리어 뒤에 다시 읽히며 모든 `BlockAtomic*` 로드는 컴파일러가 유지하는 명령입니다. 같은 커널에서 두 형식 중 하나를 사용하면 128개 스레드 모두 새 값을 읽었습니다. 이 중 하나로 슬롯을 공유하거나 매 배리어 뒤에 원시 포인터로 읽으세요. `SharedArray`가 `UnsafeCell`을 감싸는 이유도 “선언 규칙”의 팁에서 설명합니다.

---

## 공유 메모리와 다른 방식 비교

| 방식                      | 지연 시간        | 용량              | 프로그래머 작업량                |
| :---------------------------- | :------------- | :-------------------- | :------------------------------- |
| 전역 메모리(단순 방식)         | 약 500사이클    | GB 단위                   | 없음                             |
| L1/L2 캐시(암시적)        | 약 30~100사이클 | 128KB~40MB          | 없음                             |
| **공유 메모리(명시적)**  | **약 5사이클**  | **SM당 48~228KB**  | **타일링, 동기화, 뱅크 고려** |
| 레지스터                     | 약 1사이클       | SM당 64K × 32비트   | 컴파일러가 관리                 |

캐시만으로 부족할 때 프로그래머가 사용할 수 있는 도구가 공유 메모리입니다. L1과 L2 캐시는 자동으로 도움을 주지만 접근 패턴과 축출 정책에 좌우됩니다. 공유 메모리를 사용하면 무엇을 언제 불러오고 얼마나 오래 유지할지 명시적으로 제어할 수 있습니다.

---

## 뱅크 충돌

공유 메모리는 32개의 **뱅크**(워프 레인마다 하나)로 나뉩니다. 같은 워프의 두 스레드가 동일한 뱅크에 매핑되는 서로 다른 주소에 접근하면 요청이 직렬화됩니다. 이를 **뱅크 충돌**이라고 합니다. 2-way 충돌은 지연 시간이 2배, 32-way 충돌은 최대 32배가 됩니다.

매핑은 단순합니다. 연속된 32비트 워드는 연속된 뱅크에 매핑됩니다. 따라서 `TILE[0]`은 뱅크 0, `TILE[1]`은 뱅크 1에 있으며, `TILE[32]`는 다시 뱅크 0에 매핑됩니다. 흔히 쓰는 충돌 없는 접근 패턴은 다음과 같습니다.

```rust
// Each thread reads a different column: thread k reads TILE[row + k]
// If TILE_WIDTH = 32 (or a multiple), add padding: SharedArray<f32, 33 * 16>
```

16×16 타일 GEMM의 내부 루프에는 뱅크 충돌이 없습니다. `TILE_A[ty * 16 + i]`는 행을 읽으므로 연속 원소가 서로 다른 뱅크에 있고, `TILE_B[i * 16 + tx]`는 간격 16으로 열을 읽기 때문입니다.

:::{seealso}
뱅크 충돌 규칙과 패딩 전략의 상세 내용은 [CUDA 프로그래밍 가이드 — 공유 메모리 뱅크 충돌](https://docs.nvidia.com/cuda/cuda-programming-guide/#shared-memory-5-x)을 참조하세요.
:::

---

## 전체 내용 정리

MLP 파이프라인의 GEMM을 단순 방식에서 타일 방식으로 개선한 결과는 다음과 같습니다.

| 버전                   | 스레드당 전역 로드(64×64) | 스레드당 공유 메모리 로드 | 속도 향상  |
| :------------------------ | :------------------------------ | :---------------------- | :------- |
| 단순 방식(`sgemm_naive`)     | 128                             | 0                       | 1배       |
| 타일 방식(16×16 타일)       | 8 (4회 반복 × 2개 타일)      | 128                     | 약 4~10배   |
| 타일 방식 + 더블 버퍼링  | 8                               | 128 (중첩)        | 약 6~15배   |

타일 방식은 전역 메모리 대역폭을 공유 메모리 대역폭과 연산으로 교환합니다. 커널이 메모리 병목일 때 유리한 교환입니다. 더블 버퍼 방식은 다음 타일을 불러오는 동안 현재 타일을 계산해 지연 시간을 더 숨기지만, 행렬마다 `SharedArray` 두 개와 더 복잡한 동기화가 필요합니다.

:::{seealso}
- [워프 수준 프로그래밍](warp-level-programming.md) — 작은 축약에서 공유 메모리 대신 셔플 기반 축약 사용
- [텐서 메모리 가속기](tensor-memory-accelerator.md) — 수동 로드 루프를 대체하는 하드웨어 가속 전역→공유 메모리 복사
:::

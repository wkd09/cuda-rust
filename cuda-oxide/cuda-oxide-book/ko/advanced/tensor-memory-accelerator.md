# 텐서 메모리 가속기 (TMA)

Hopper 아키텍처(SM 90)부터 NVIDIA GPU에는 전역 메모리와 공유 메모리 사이의 데이터를 스레드 실행 리소스를 *점유하지 않고* 옮기는 전용 하드웨어 장치인 **텐서 메모리 가속기**가 포함됩니다. 블록의 모든 스레드가 각자 로드 명령을 내리는 대신 스레드 하나가 설명자를 TMA 엔진에 전달하고 하드웨어가 전송 전체를 비동기로 처리합니다. 나머지 127개 스레드는 계산하거나 다른 데이터를 불러오거나 배리어에서 기다릴 수 있습니다.

cuda-oxide는 `TmaDescriptor`, `cp_async_bulk_tensor_*` 계열, `ManagedBarrier` 타입 상태 API를 통해 TMA를 제공합니다. 이 장에서는 설정과 복사 패턴, 효율적인 로드/계산 파이프라인을 위한 배리어 통합 방법을 설명합니다.

:::{seealso}
하드웨어 장치와 swizzle 모드, 지원 텐서 차원의 상세 설명은 [CUDA 프로그래밍 가이드 — TMA 비동기 데이터 복사](https://docs.nvidia.com/cuda/cuda-programming-guide/#tensor-memory-access)를 참조하세요.
:::

---

## TMA가 해결하는 문제

[공유 메모리](../advanced/shared-memory-and-synchronization.md) 장에서는 블록의 모든 스레드가 전역 메모리에서 타일을 불러옵니다.

```rust
TILE_A[ty * TILE + tx] = a[row * k + tile_offset + tx];
```

이 방식은 작동하지만 블록의 모든 스레드(128, 256 또는 1024개)가 주소를 계산하고 로드 명령을 발행하는 데 시간을 씁니다. 타일이 크면 로드 단계가 커널 실행 시간을 지배할 수 있습니다.

TMA는 스레드별 로드 루프를 하드웨어 명령 하나로 대체합니다. 스레드 하나가 “전역 메모리의 이 2차원 영역을 공유 메모리로 복사하라”고 요청하면 TMA 엔진이 주소 계산, stride 처리, 뱅크 충돌 없는 swizzle 쓰기까지 나머지를 맡습니다.

```{figure} images/tma-async-pipeline.svg
:align: center
:width: 100%

TMA 비동기 대량 복사 파이프라인입니다. 호스트는 텐서 레이아웃을 담는 `TmaDescriptor`를 만듭니다. 장치에서는 스레드 하나가 복사를 요청하고 모든 스레드가 mbarrier에서 대기합니다. TMA 엔진이 비동기로 전송하고 완료 시 배리어를 신호하며, 스레드는 공유 메모리에서 계산을 이어갑니다.
```

---

## TmaDescriptor — 텐서 맵

`TmaDescriptor`는 TMA 엔진이 텐서에 대해 알아야 할 베이스 주소, 차원, 원소 타입, stride, swizzle 모드를 인코딩하는 128바이트 불투명 구조체입니다. CUDA Driver API를 사용해 **호스트**에서 만듭니다.

```rust
use cuda_device::tma::TmaDescriptor;

// On the host (simplified — the real call has many parameters)
unsafe {
    cuTensorMapEncodeTiled(
        &mut desc as *mut TmaDescriptor as *mut _,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                       // 2D tensor
        global_ptr,              // base address in global memory
        &dims,                   // [rows, cols]
        &strides,                // byte strides per dimension
        &box_dims,               // tile size per copy
        &elem_strides,           // element strides within a box
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, // bank-conflict-free swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    );
}
```

설명자는 이후 일반 인자로 커널에 전달됩니다. 장치에서는 `*const TmaDescriptor`로 취급합니다. 즉 불투명한 읽기 전용 값이며 설명 대상 할당이 살아 있는 동안 유효합니다.

:::{tip}
swizzle 모드는 성능에 가장 큰 영향을 미치는 매개변수입니다. `SWIZZLE_128B`는 공유 메모리의 바이트 레이아웃을 재배치해 서로 다른 워프의 128바이트 접근이 서로 다른 뱅크에 도달하게 합니다. swizzle이 없으면 단순한 2차원 타일 로드에서 뱅크 충돌이 자주 발생합니다.
:::

---

## 전역 메모리에서 공유 메모리로 복사 (G2S)

`cp_async_bulk_tensor_*_g2s` 함수는 전역 메모리에서 공유 메모리 대상으로 TMA 복사를 요청합니다. 1차원부터 5차원 텐서까지 지원합니다.

```rust
use cuda_device::barrier::Barrier;
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};

unsafe fn load_tile(
    smem_dst: *mut u8,
    desc: *const TmaDescriptor,
    tile_x: i32,
    tile_y: i32,
    bar: *mut Barrier,
) {
    cp_async_bulk_tensor_2d_g2s(smem_dst, desc, tile_x, tile_y, bar);
}
```

좌표(`tile_x`, `tile_y`)는 설명자에 인코딩된 box 크기 단위로 복사할 타일을 지정합니다. 배리어 포인터는 TMA 엔진이 완료 신호를 보낼 위치를 알려줍니다.

이 함수들은 클러스터 공유 메모리를 대상으로 합니다. 대상은 클러스터의 어느 CTA에나 속할 수 있으므로 컴파일러가 클러스터 윈도우로 변환합니다. 복사 대상이 요청한 CTA 자체의 공유 메모리인 경우 `cp_async_bulk_tensor_*_g2s_cta`는 변환 없이 `.shared::cta` 대상 형식(PTX 8.6, sm_90 이상)을 생성합니다. 대상 타입은 CTA 공유 포인터인 [`SharedPtr<u8>`](../advanced/shared-memory-and-synchronization.md)이므로 도우미 함수를 통해 전달하는 경우에도 어느 경로에서 포인터를 얻었는지와 무관하게 로컬성이 보장됩니다.

```rust
// SAFETY: TILE and BAR are `static mut` shared-memory objects of this kernel,
// and the copy completes on BAR before any thread reads TILE.
unsafe {
    cp_async_bulk_tensor_2d_g2s_cta(
        SharedArray::shared_ptr(&raw mut TILE).cast::<u8>(),
        tensor_map, tile_x, tile_y,
        &raw mut BAR,
    );
}
```

SM12x에서는 이 형식이 성능에 영향을 줍니다. ptxas는 CTA 로컬임을 증명할 수 없는 비 멀티캐스트 `.shared::cluster` TMA 로드에 런타임 검사와 시스템 호출을 넣습니다. 그 결과 `setmaxnreg`도 무시합니다.

```{figure} images/tma-cta-local.svg
:align: center
:width: 100%

두 G2S 대상 형식의 비교입니다. 왼쪽의 클러스터 형식은 일반 `*mut u8`을 클러스터 윈도우로 변환합니다. sm_120a에서 ptxas는 런타임 CTA 검사를 추가하고, 그 호출 때문에 `setmaxnreg`를 무시합니다. 가운데의 `SharedPtr<T>`는 도우미 함수 매개변수, 구조체 필드, 반환값을 거쳐도 타입에 공유 주소 공간을 유지합니다. 오른쪽의 `cp_async_bulk_tensor_2d_g2s_cta`는 `SharedPtr<u8>`를 받아 주소 변환 없이 `.shared::cta` 형식을 생성하므로 커널이 `setmaxnreg`와 레지스터 168개 설정을 유지할 수 있습니다.
```

### 복사는 스레드 하나만 요청

TMA 복사는 블록 전체가 함께 수행하는 **집합 연산이 아닙니다**. `cp_async_bulk_tensor_*_g2s`는 **정확히 스레드 하나**만 호출해야 합니다. 나머지 스레드는 배리어 메커니즘으로 참여합니다.

```rust
if thread::threadIdx_x() == 0 {
    let token = bar.arrive_expect_tx(tile_bytes as u32);
    unsafe {
        cp_async_bulk_tensor_2d_g2s(dst, desc, x, y, bar.as_ptr() as *mut _);
    }
    bar.wait(token);
} else {
    let token = bar.arrive();
    bar.wait(token);
}
```

`arrive_expect_tx(bytes)`는 스레드 도착 외에 `bytes` 바이트만큼의 트랜잭션 완료를 더 기다리도록 배리어에 알립니다. 모든 스레드가 도착하고 TMA 엔진이 기대한 바이트를 모두 전달해야 배리어가 열립니다.

---

## mbarrier 사용 순서

TMA 완료 추적은 `ManagedBarrier`(또는 원시 `mbarrier_*` 함수)에 의존합니다. 타입 상태 API가 생명 주기를 타입 수준에서 보장합니다.

```rust
use cuda_device::SharedArray;
use cuda_device::barrier::{Barrier, ManagedBarrier, Ready, TmaBarrierHandle, Uninit};

#[kernel]
pub fn tma_load_kernel(desc: *const TmaDescriptor) {
    static mut BAR: SharedArray<Barrier, 1, 128> = SharedArray::UNINIT;

    let bar: TmaBarrierHandle<Ready> = unsafe {
        TmaBarrierHandle::<Uninit>::from_static(BAR.as_mut_ptr())
            .init_by(block_size, 0) // thread 0 inits, includes fence + sync
    };

    // Issue TMA copy
    if thread::threadIdx_x() == 0 {
        let token = bar.arrive_expect_tx(TILE_BYTES as u32);
        unsafe {
            cp_async_bulk_tensor_2d_g2s(
                smem_ptr, desc, tile_x, tile_y, bar.as_ptr() as *mut _
            );
        }
        bar.wait(token);
    } else {
        let token = bar.arrive();
        bar.wait(token);
    }

    // Shared memory now contains the tile — safe to read
    // ...

    unsafe { bar.inval(); }
}
```

생명 주기는 `Uninit` → `init()` → `Ready` → `arrive()` / `wait()` → `inval()` → `Invalidated`입니다. 타입 시스템은 초기화되지 않은 배리어에서 `wait`를 호출하거나 무효화된 배리어에서 `arrive`를 호출하지 못하게 합니다.

```{figure} images/mbarrier-thread-election.svg
:align: center
:width: 100%

`mbarrier.init`을 어느 스레드가 실행하는지 보여줍니다. 모든 스레드가 `init_by(count, id)`와 `inval_by(id)`를 호출하지만, 블록 선형 ID(`x + blockDim.x * (y + blockDim.y * z)`, `thread_rank()`가 사용하는 번호)가 `id`와 같은 스레드만 명령을 실행합니다. `init()`과 `inval()`은 ID 0, 즉 스레드 (0, 0, 0)을 지정합니다. `threadIdx.x`만 비교하면 1차원 블록에서만 스레드 하나를 지정합니다. (32, 4, 1) 블록에서는 선택자 0이 네 스레드에서 실행되고 선택자 127은 아무 스레드에서도 실행되지 않아 실행이 오류 719로 실패합니다.
```

### 주요 배리어 규칙

| 규칙 | 이유 |
|:---|:---|
| `init_by`가 `fence_proxy_async_shared_cta()`를 포함 | TMA 엔진이 공유 메모리에서 초기화된 배리어를 확인해야 합니다. |
| 복사 전 `arrive_expect_tx` 호출 | 엔진이 쓰기를 시작하기 전에 기대 바이트 수를 알아야 합니다. |
| 모든 스레드가 도착하거나 도착한 것으로 계산되어야 함 | 기대된 스레드 도착 수와 TX 바이트가 충족되어야 배리어가 열립니다. |
| 재사용하지 않으면 완료 후 `inval()` 호출 | 배리어 하드웨어 리소스를 반환합니다. |

---

## 공유 메모리에서 전역 메모리로 복사 (S2G)

TMA는 반대 방향도 지원합니다. 이 패턴에서는 배리어 대신 커밋 그룹을 사용합니다.

```rust
use cuda_device::tma::{
    cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_s2g, cp_async_bulk_wait_group,
};

unsafe {
    cp_async_bulk_tensor_2d_s2g(smem_src, desc, tile_x, tile_y);
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group(0); // wait for all outstanding groups
}
```

S2G 복사에는 배리어를 사용하지 않습니다. 대신 `commit_group`이 대기 중인 복사들을 그룹으로 묶고 `wait_group(n)`은 진행 중인 그룹이 최대 `n`개 남을 때까지 기다립니다. `wait_group(0)`은 모든 작업이 끝날 때까지 대기합니다.

---

## 정렬 요구사항

공유 메모리의 TMA 대상은 **128바이트 정렬**이어야 합니다. cuda-oxide는 `SharedArray`와 `DynamicSharedArray`의 `ALIGN` 매개변수로 이를 제공합니다.

```rust
static mut TILE: SharedArray<f32, 1024, 128> = SharedArray::UNINIT;

// Or with dynamic shared memory:
let dst: *mut u8 = DynamicSharedArray::<u8, 128>::get();
```

대상이 128바이트 정렬되지 않으면 TMA 엔진은 잘못된 결과를 조용히 내거나 오류를 발생시킵니다. 런타임 검사는 없으므로 정렬이 구성 시점부터 올바르게 보장되어야 합니다.

---

## 멀티캐스트 변형

클러스터를 사용하는 Hopper 이상에서 TMA는 전역 메모리 로드 하나를 여러 CTA의 공유 메모리에 동시에 멀티캐스트할 수 있습니다.

```rust
use cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast;

unsafe {
    cp_async_bulk_tensor_2d_g2s_multicast(
        dst, desc, x, y, bar_ptr,
        cta_mask,  // u16 bitmask of target CTAs in the cluster
    );
}
```

이는 여러 스레드 블록이 같은 A 또는 B 타일을 필요로 하는 GEMM 계열 커널에 특히 유용합니다. 각 블록이 복사본을 따로 로드하는 대신 TMA 작업 하나가 모두에게 전달합니다.

:::{seealso}
TMA 멀티캐스트와 자연스럽게 결합하는 클러스터 모델 및 분산 공유 메모리는 [클러스터 프로그래밍](cluster-programming.md)을 참조하세요.
:::

---

## TMA와 수동 로드 비교

| 속성 | 스레드별 수동 로드 | TMA 대량 복사 |
|:---|:---|:---|
| 필요한 스레드 | 블록의 모든 스레드 | 스레드 하나가 요청, 하드웨어가 실행 |
| 주소 계산 | 스레드마다 인덱스 계산 | 설명자에 인코딩 |
| 뱅크 충돌을 위한 swizzle | 직접 패딩 | 설명자 수준의 swizzle 모드 |
| 완료 추적 | `sync_threads()` | TX 추적을 사용하는 `mbarrier` |
| 계산과 중첩 | 스레드가 로드 중이므로 불가 | 다음 타일을 불러오는 동안 이전 타일 계산 |
| 최소 아키텍처 | 모든 CUDA GPU | Hopper (SM 90 이상) |

TMA의 핵심 장점은 최대 대역폭 자체가 아닙니다. TMA와 수동 로드는 비슷한 최대 처리량을 낼 수 있습니다. 차이는 **스레드 활용률**입니다. TMA가 다음 타일을 로드하는 동안 모든 스레드가 현재 타일을 계산할 수 있습니다. 이 방식은 최신 GEMM 구현이 사용하는 **다단계 소프트웨어 파이프라인**을 가능하게 합니다.

:::{seealso}
- TMA가 대체하는 수동 로드 패턴은 [공유 메모리와 동기화](../advanced/shared-memory-and-synchronization.md)를 참조하세요.
- TMA가 텐서 코어에 데이터를 공급하는 방법은 [행렬 곱 가속기](matrix-multiply-accelerators)를 참조하세요.
:::

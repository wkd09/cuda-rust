# 클러스터 프로그래밍

**스레드 블록 클러스터(thread block cluster)**는 같은 GPC(Graphics Processing Cluster)에 있는 SM에서 동시에 실행되도록 보장된 스레드 블록의 모음입니다. Hopper(SM 90)에서 도입된 클러스터는 표준 CUDA에서 제공하지 않던 기능, 즉 전역 메모리를 거치지 않는 **블록 간 공유 메모리 접근**을 제공합니다.

일반적인 CUDA에서는 스레드 블록이 서로 독립적이므로 블록 0이 블록 1의 공유 메모리를 읽을 수 없습니다. 클러스터에서는 하드웨어가 모든 구성 블록의 공유 메모리를 하나의 **분산 공유 메모리(DSMEM)** 주소 공간에 매핑합니다. 클러스터의 스레드는 어느 블록의 공유 메모리든 직접 읽을 수 있으며, 전역 메모리를 왕복하지 않고 공유 메모리 수준의 지연 시간으로 접근합니다.

cuda-oxide는 `cuda_device::cluster` 모듈과 `#[cluster_launch]` 속성을 통해 클러스터 기능을 제공합니다. 이 장에서는 프로그래밍 모델, DSMEM 접근 방식, 고성능 행렬 커널에서 클러스터와 TMA 멀티캐스트를 조합하는 방법을 설명합니다.

:::{seealso}
하드웨어 사양, GPC 제약, 최대 클러스터 크기는 [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/#thread-block-clusters)를 참고하세요.
:::

---

## 클러스터 모델

```{figure} images/cluster-dsmem-topology.svg
:align: center
:width: 100%

4개 블록으로 구성된 클러스터. 각 블록에는 자체 공유 메모리가 있지만 하드웨어는 네 영역을 하나의 DSMEM 공간에 매핑합니다. `cluster_sync()` 배리어 이후 각 블록은 다른 블록의 공유 메모리를 읽을 수 있습니다. 점선 화살표는 블록 간 DSMEM 읽기를 나타냅니다.
```

클러스터는 커널을 시작할 때 지정합니다. 하드웨어는 클러스터 안의 모든 블록을 같은 GPC의 SM에 배치해 지연 시간이 짧은 블록 간 통신을 보장합니다. 최대 클러스터 크기는 아키텍처에 따라 달라집니다. Hopper에서는 클러스터당 최대 8개 블록을 사용할 수 있습니다.

### 클러스터 커널 선언

cuda-oxide에서는 커널에 `#[cluster_launch]`를 붙입니다.

```rust
use cuda_device::{SharedArray, cluster, cluster_launch, kernel, thread};

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(/* ... */) {
    // 이 커널은 X 차원에 블록 4개가 있는 클러스터로 실행됩니다.
    let rank = cluster::block_rank();
    let size = cluster::cluster_size();
    // ...
}
```

`#[cluster_launch(x, y, z)]` 속성은 다음 작업을 수행합니다.

1. PTX의 `.reqnctapercluster` 지시문을 설정하는 `__cluster_config<X, Y, Z>()` 함수를 삽입합니다.
2. 호스트 쪽 실행에서는 이에 맞는 `cluster_dim` 매개변수와 함께 `launch_kernel_ex`를 사용해야 합니다.

---

## 클러스터 내 위치 확인

모든 스레드는 클러스터 계층 안에서 자신의 위치를 조회할 수 있습니다.

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();       // 현재 클러스터에서 0..cluster_size()-1
let size = cluster::cluster_size();     // 클러스터 전체 블록 수
let cidx = cluster::cluster_idx();      // 그리드에서 현재 클러스터의 번호
let ncls = cluster::num_clusters();     // 그리드의 전체 클러스터 수
```

`block_rank()`는 DSMEM 접근에 사용하는 핵심 식별자입니다. 어느 블록의 공유 메모리를 읽을지 알려줍니다. 클러스터 안에서 블록마다 부여된 로컬 디바이스 ID라고 생각하면 됩니다.

3차원 클러스터 위치 전체를 확인하려면 다음 함수를 사용합니다.

```rust
let cx = cluster::cluster_ctaidX();     // 클러스터 안에서 블록의 X 위치
let cy = cluster::cluster_ctaidY();     // 클러스터 안에서 블록의 Y 위치
let cz = cluster::cluster_ctaidZ();     // 클러스터 안에서 블록의 Z 위치
```

---

## 분산 공유 메모리(DSMEM)

클러스터의 핵심 기능은 블록 간 공유 메모리 접근입니다. 다른 블록의 공유 메모리를 읽는 방법은 두 가지입니다.

### 방법 1: `map_shared_rank`로 매핑한 뒤 일반 역참조 사용

`map_shared_rank`는 로컬 공유 메모리 포인터를 받아 다른 블록의 공유 메모리에서 같은 오프셋을 가리키는 실제 포인터를 돌려줍니다. 이 포인터는 클러스터 공유 주소 공간을 가리키므로 일반 포인터처럼 역참조할 수 있습니다.

```rust
use cuda_device::{SharedArray, cluster, thread};

static mut DATA: SharedArray<u32, 256> = SharedArray::UNINIT;

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn halo_exchange(/* ... */) {
    let tid = thread::threadIdx_x() as usize;
    let rank = cluster::block_rank();

    // 각 블록은 자기 공유 메모리에 값을 씁니다.
    unsafe { DATA[tid] = compute_value(rank, tid); }

    thread::sync_threads();    // 먼저 블록 내부 배리어
    cluster::cluster_sync();   // 다음으로 클러스터 전체 배리어

    // 다음 블록의 공유 메모리에서 읽습니다.
    let neighbor = (rank + 1) % cluster::cluster_size();
    let remote_ptr = unsafe {
        cluster::map_shared_rank(DATA.as_ptr().add(tid), neighbor)
    };
    let neighbor_val = unsafe { *remote_ptr };
}
```

마지막 줄은 일반 Rust 읽기입니다. 컴파일러는 매핑된 포인터가 클러스터 공유 주소 공간을 가리킨다는 점을 유지하므로 `ld.shared::cluster` 명령으로 변환합니다. 쓰기도 같은 방식으로 동작합니다. `map_shared_rank_mut`로 포인터를 매핑해 값을 할당하면 `st.shared::cluster` 저장 명령이 생성됩니다. 이 방법은 모든 타입에서 사용할 수 있으며 `sm_90+`가 필요합니다.

### 방법 2: `dsmem_read_u32`로 고정 폭 `u32` 읽기

`dsmem_read_u32`는 `u32` 하나를 매핑하고 읽는 작업을 한 번에 수행합니다. 로컬 포인터와 대상 rank를 전달하면 함수가 rank 매핑을 직접 처리합니다.

```rust
let neighbor_val = unsafe {
    cluster::dsmem_read_u32(
        DATA.as_ptr() as *const u32,
        neighbor,  // 대상 rank
    )
};
```

두 방법 모두 같은 `ld.shared::cluster` PTX 명령을 생성합니다. 일반적으로는 모든 타입에서 동작하고 쓰기도 지원하는 `map_shared_rank` 후 역참조 방식을 사용하세요. `u32` 하나만 읽고 매핑과 로드를 한 번의 호출로 처리하고 싶을 때 `dsmem_read_u32`를 선택하면 됩니다.

---

## 동기화

클러스터는 블록 동기화와 전역 동기화 사이에 새로운 동기화 범위를 추가합니다.

| 프리미티브 | 범위 | 사용 시점 |
| :--- | :--- | :--- |
| `thread::sync_threads()` | 블록 | 한 블록 안의 스레드끼리 동기화할 때 |
| `cluster::cluster_sync()` | 클러스터 | 클러스터 안의 모든 블록을 동기화할 때 |
| `mbarrier_arrive_cluster(addr)` | 클러스터 | 다른 블록에 있는 배리어에 도착을 알릴 때 |

DSMEM에 접근하기 전에 지켜야 할 동기화 순서는 다음과 같습니다.

1. 로컬 공유 메모리에 **쓰기**
2. **`sync_threads()`** — 블록의 모든 스레드가 쓰기를 마쳤는지 확인
3. **`cluster_sync()`** — 클러스터의 모든 블록이 이 지점까지 도달했는지 확인
4. DSMEM을 통해 원격 공유 메모리에서 **읽기**

둘 중 하나라도 빠지면 데이터 경쟁이 발생합니다. `cluster_sync()` 없이 `sync_threads()`만 호출하면 현재 블록은 준비됐지만 이웃 블록은 아직 준비되지 않았을 수 있습니다. 반대로 `sync_threads()` 없이 `cluster_sync()`만 호출하면 클러스터는 동기화되어도 현재 블록의 쓰기가 아직 보이지 않을 수 있습니다.

---

## 클러스터를 사용한 TMA 멀티캐스트

클러스터는 TMA의 강력한 기능 중 하나인 **멀티캐스트 복사**를 지원합니다. TMA 로드 한 번으로 동일한 타일을 여러 블록의 공유 메모리에 동시에 쓸 수 있습니다.

```rust
use cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast;

// CTA 마스크: 비트 0..3을 설정해 클러스터의 블록 4개 모두에 타일 전달
let cta_mask: u16 = 0b1111;

unsafe {
    cp_async_bulk_tensor_2d_g2s_multicast(
        smem_dst, desc, tile_x, tile_y, bar_ptr, cta_mask
    );
}
```

멀티캐스트가 없으면 각 블록이 TMA 복사를 따로 요청하므로 전역 메모리를 네 번 읽습니다. 멀티캐스트를 사용하면 TMA 엔진이 데이터를 한 번 읽어 블록 네 개에 분배합니다. 클러스터의 모든 블록에서 같은 A 또는 B 타일을 사용하는 GEMM 커널에서 특히 유용합니다.

`cta_mask`는 각 비트가 어느 rank에 복사본을 전달할지 나타내는 비트마스크입니다. 클러스터의 일부 블록에만 선택적으로 멀티캐스트할 수도 있습니다.

---

## 실제 예제: 헤일로 교환

클러스터의 흔한 활용 사례는 스텐실 계산의 **헤일로 교환**입니다. 각 블록이 격자 타일을 처리하며 이웃 블록의 경계 원소를 필요로 합니다. 클러스터가 없다면 전역 메모리에 쓰고 읽거나 스트림 동기화를 신중히 구성해야 합니다. 클러스터를 사용하면 블록 안에서 가까운 메모리 접근으로 처리할 수 있습니다.

```rust
use cuda_device::{DisjointSlice, SharedArray, cluster, cluster_launch, kernel, thread};

const TILE_W: usize = 256;
const HALO: usize = 1;
const SMEM_W: usize = TILE_W + 2 * HALO;

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn stencil_1d(input: &[f32], mut output: DisjointSlice<f32>, n: u32) {
    static mut SMEM: SharedArray<f32, { SMEM_W }> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let rank = cluster::block_rank();
    let global_idx = rank as usize * TILE_W + tid;

    // 내부 영역을 읽습니다. HALO만큼 이동해 헤일로 슬롯을 비워 둡니다.
    unsafe {
        if global_idx < n as usize {
            SMEM[tid + HALO] = input[global_idx];
        }
    }

    thread::sync_threads();
    cluster::cluster_sync();

    // 이전 블록에서 왼쪽 헤일로를 읽습니다.
    if tid == 0 && rank > 0 {
        let prev_rank = rank - 1;
        unsafe {
            SMEM[0] = f32::from_bits(cluster::dsmem_read_u32(
                SMEM.as_ptr().add(TILE_W) as *const u32, prev_rank
            ));
        }
    }

    // 다음 블록에서 오른쪽 헤일로를 읽습니다.
    if tid == 0 && rank < cluster::cluster_size() - 1 {
        let next_rank = rank + 1;
        unsafe {
            SMEM[TILE_W + HALO] = f32::from_bits(cluster::dsmem_read_u32(
                SMEM.as_ptr().add(HALO) as *const u32, next_rank
            ));
        }
    }

    thread::sync_threads();

    // 3점 스텐실: output[i] = 0.25 * left + 0.5 * center + 0.25 * right
    if global_idx < n as usize {
        let left = unsafe { SMEM[tid + HALO - 1] };
        let center = unsafe { SMEM[tid + HALO] };
        let right = unsafe { SMEM[tid + HALO + 1] };

        unsafe {
            *output.get_unchecked_mut(global_idx) = 0.25 * left + 0.5 * center + 0.25 * right;
        }
    }
}
```

클러스터가 없다면 헤일로 교환은 경계 원소를 전역 메모리에 쓰고, 이벤트나 스트림으로 동기화한 뒤 다시 읽어야 합니다. 클러스터에서는 `cluster_sync()`와 `dsmem_read_u32`만으로 끝나므로 공유 메모리 수준의 지연 시간으로 처리하고 전역 메모리를 사용하지 않습니다.

---

## 제약과 권장 사항

| 제약 | 세부 내용 |
| :--- | :--- |
| 최대 클러스터 크기 | 아키텍처에 따라 블록 최대 8개 |
| 스케줄링 보장 | 클러스터의 모든 블록은 같은 GPC에서 동시에 실행 |
| DSMEM 지연 시간 | 로컬 공유 메모리와 유사(약 5~10 사이클) |
| 클러스터 차원 선언 | `#[cluster_launch(x, y, z)]`와 호스트 실행 설정의 `cluster_dim`이 필요 |
| 블록 수 배수 조건 | 그리드의 각 차원 블록 수는 클러스터 크기의 배수여야 함 |
| 클러스터/비클러스터 혼용 | 같은 커널에서는 지원하지 않음 |

:::{tip}
클러스터를 사용하면 하드웨어가 모든 블록을 같은 GPC에 배치해야 합니다. 클러스터 크기가 GPC 용량에 비해 너무 크면 점유율이 낮아질 수 있습니다. 먼저 2~4개 블록으로 시작해 측정한 다음 규모를 늘리세요.
:::

:::{seealso}
- [Tensor Memory Accelerator](https://nvidia.github.io/cuda-rust/cuda-oxide/advanced/tensor-memory-accelerator.html) — TMA 멀티캐스트는 CTA 사이로 데이터를 전달할 때 클러스터를 필요로 합니다.
- [Matrix Multiply Accelerators](https://nvidia.github.io/cuda-rust/cuda-oxide/advanced/matrix-multiply-accelerators.html) — CG2 모드는 클러스터 쌍을 사용해 더 넓은 MMA 타일을 처리합니다.
- [Shared Memory and Synchronization](https://nvidia.github.io/cuda-rust/cuda-oxide/advanced/shared-memory-and-synchronization.html) — DSMEM이 확장하는 블록 단위 공유 메모리의 기초입니다.
:::

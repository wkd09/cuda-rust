# API 빠른 참조

이 부록은 cuda-oxide 장치 및 호스트 API에 대한 압축된 참조입니다. 전체 문서를 위해 작업 공간 루트에서 `cargo doc --no-deps --open`를 실행하십시오.

---

## 속성 및 매크로

### 커널 및 장치 속성

```rust
use cuda_device::{cluster_launch, cooperative_launch, device, kernel, launch_bounds};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { /* ... */ }

#[kernel]
pub fn unrolled(mut data: DisjointSlice<f32>) {
    let mut i = 0;
    #[unroll(4)]
    while i < 16 { /* ... */ i += 1; }
}

#[kernel]
#[launch_bounds(256, 2)]
pub fn tuned_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cooperative_launch]
pub fn grid_sync_kernel(data: &mut [f32]) { /* ... */ }

#[device]
fn helper(x: f32) -> f32 { x * x }
```

 속성                                   | 목적                                                             
|:--------------------------------------------|:--------------------------------------------------------------------|
 `#[cuda_module]`                            | 모듈의 커널을 `load` 및 런처로 타입화된 호스트 모듈로 수집합니다 
 `#[kernel]`                                 | 함수를 GPU 커널 진입점로 표시 (PTX에서 `.entry`)       
 `#[device]`                                 | 보조 기능을 표시하거나`extern "C"`장치 컴파일 블록 
 `#[unroll]` / `#[unroll(N)]`               | 전체적인 배열을 요청하거나 `N >= 2`의 배열            
 `#[launch_bounds(max_threads, min_blocks)]` | 레지스터 할당에 대한 점유율 팁                             
 `#[constant]`                               | `ConstantMemory<T>`를 고정 메모리에 위치하여 호스트 `set_<name>`를 사용하십시오. 
 `#[launch_contract(...)]`                   | 커널이 필요로 하는 시작 형태를 선언하여 안전한 (비 `unsafe`) 시작을 잠금 해제합니다 
 `#[cluster_launch(x, y, z)]`                | 컴파일 타임 클러스터 크기를 설정 (Hopper+)                       
 `#[cooperative_launch]`                     | `#[cuda_module]`를 통해 협동으로 시작 (`grid::sync()`를 활성화)  
 `#[convergent]`                             | 컨버전트 (장벽 의미) 로 표시                              
 `#[pure]`                                   | 부작용이 없는 것으로 표시하세요                                            
 `#[readonly]`                               | 읽기 전용으로 표시하세요                                                   

이 주석을 명시적으로 계산된 경우에만 사용하세요`while`루프 또는 범위`for`루프 (`for i in a..b`또는`(a..b).rev()`, 을 포함한 모든 정수 카운터 위에`isize`) 안쪽은`#[kernel]`또는`#[device]`함수. 다른 반복 루프는 아직 풀림 패스에 의해 인식되지 않습니다. 둥글게 된 루프와 여러 `continue`경로 지원 됩니다.`#[unroll]``break`경로와 여러 출구 목표물

부분 `#[unroll(N)]`는 긍정적 단계, `<` 또는 `<=` 테스트, 변하지 않는 한도, 그리고 정상적인 헤더 테스트 이외의 출구가 필요하지 않습니다. 다른 요청은 경고되며 풀리지 않습니다.

한 개의 주석은 최대 1,024개의 본체 복사본, 8,192개의 복제된 기본 블록, 65,536개의 복식된 연산을 생성할 수 있다. 더 큰 요청은 경고를 받고 풀리지 않는다.

### 디버그 및 PTX 매크로

```rust
use cuda_device::{gpu_assert, gpu_printf, ptx_asm};

gpu_printf!("thread {}: val = {}\n", idx, val);
gpu_assert!(val.is_finite());
gpu_assert!(val >= 0.0, "expected non-negative value");

let y: u32;
unsafe {
    ptx_asm!("add.u32 %0, %1, %1;", out("=r") y, in("r") x, options(register_only));
}
```

 매크로                               | 목적                                                           
|:------------------------------------|:------------------------------------------------------------------|
 `gpu_printf!(fmt, args...)`         | 장치 측 포맷 출력 (<`vprintf`>)                
 `gpu_assert!(condition)`            | 실행 시간 확인; 호출`trap()`실패 경우                      
 `gpu_assert!(condition, "message")` | CUDA 진단과 실행 시간 주장; 메시지는 문자이어야 합니다 
 `ptx_asm!(...)`                     | unsafe CUDA 인라인 PTX                                            

메시지 양식은 CUDA의 장치 측 `__assertfail` 시스템 호출에 내려갑니다. 드라이버는 메시지 및 호출 사이트 메타 데이터를보고하고 동기화로 `CUDA_ERROR_ASSERT`를 반환합니다.

---

## 컴파일 타임 정책 구성

```rust
use cuda_device::config::{
    Atom, AtomKind, AtomSpec, Block, Cluster, ColumnMajor, Global, Layout, MemorySpace, Policy,
    PolicyId, Register, RowMajor, Scope, Shape, Shape1, Shape2, Shape3, Shared, TensorMemory,
    Thread, Tile, TileSpec, Warp, WarpGroup,
};
```

정책은 제로 크기의 러스트 타입을 사용하여 컴파일 타임 커널 구성을 설명합니다. 일반 커널은 각 구체적인 정책 유형에 대해 한 번 단칭됩니다. 정책은 실행 시간 커널 인수로 전달되지 않습니다.

```rust
trait VectorPolicy: Policy {
    type BlockTile: TileSpec;
    type ElementAtom: AtomSpec;

    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum SmallTilePolicy {}

impl Policy for SmallTilePolicy {
    const ID: PolicyId =
        PolicyId::new(0x706f_6c69_6379_5f63, 1);
}
```

 API                  | 설명                                                                             
| :------------------- |:----------------------------------------------------------------------------------------|
 `Policy`             | 이름 붙인 컴파일 타임 커널 정책의 최소 기본 특질                               
 `PolicyId`           | 프로젝트별 네임스페이스와 정책 로컬 값을 포함하는 명시적 안정적인 아이덴티티 
 `Shape`              | 정적 형상의 순위, 확장 및 확인된 요소 수를 노실행키는 특성                
 `Shape1<D0>`         | 1차원 컴파일 타임 형상                                                      
 `Shape2<D0, D1>`     | 2차원 컴파일 타임 형상                                                      
 `Shape3<D0, D1, D2>` | 3차원 컴파일 타임 형상                                                    
 `Tile<S, L, M, Q>`   | 형태, 레이아웃, 메모리 공간 및 실행 범위를 결합하는 메타 데이터 만의 설명    
 `TileSpec`           | 타일 설명의 구성 요소에 대한 타입 레벨 접근                               
 `Atom<K, S, Q>`      | 메타데이터만 포함된 작업 설명, 논리 발자국 및 참여 스레드 
 `AtomKind`           | 도메인 특정 동작을 식별하는 오픈 마커 특성                               
 `AtomSpec`           | 원자의 동작 종류, 형태, 범위에 대한 타입 레벨 접근                         
 `Layout`             | 메모리 순서 메타데이터에 대한 오픈 트리트                                                    
 `RowMajor`           | 가장 오른쪽 좌표가 과 접해 있는 레이아웃                                         
 `ColumnMajor`        | 좌우 좌표가 과 접해 있는 레이아웃                                          
 `MemorySpace`        | CUDA 저장 위치를 설명하는 오픈 트리트                                           
 `Global`             | 장치 전역 메모리 마커                                                             
 `Shared`             | 블록별 공유 메모리 마커                                                          
 `Register`           | 스레드 로컬 레지스터 저장 표시기                                                    
 `TensorMemory`       | 하드웨어 텐서 메모리 마커                                                           
 `Scope`              | 오픈 트레이트: 작업에 협력하는 스레드를 설명하는 특징                           
 `Thread`             | 단일 스레드 실행 범위                                                           
 `Warp`               | 단발전 실행 범위                                                             
 `WarpGroup`          | 하드웨어 워크그룹 실행 범위                                                      
 `Block`              | 스레드 블록 실행 범위                                                            
 `Cluster`            | 스레드 블록 클러스터 실행 범위                                                    

`Tile` 및 `Atom`는 설명일 뿐. 저장소를 할당하거나 포인터 액세스를 제공하거나 GPU 명령어를 발산하거나 스레드를 동기화하거나 보안 속성을 설정하지 않습니다. 도메인 특정 정책 특성은 해당 설명자에 의미를 부여하고 지원되는 조합을 검증합니다.

`PolicyId`값은 정책 라이브러리에서 명시적으로 제공됩니다. 그들은 강성에서 파생되지 않습니다.`TypeId`, 타입 이름, 컴파일러 망글링 또는 해시. 정책의 생성된 동작이 변하지 않는 동안 ID를 안정적으로 유지하고 그 동작이 변경되면 새로운 값을 할당합니다.

정책 관련 상수는 현재 컴파일 시간 `launch_bounds` 및 부분 루프 `unroll` 표현식에서 사용할 수 있습니다:

```rust
use cuda_device::{kernel, launch_bounds};

#[kernel]
#[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
pub unsafe fn transform<P: VectorPolicy>(
    input: *const u32,
    output: *mut u32,
    count: u32,
) {
    let mut lane = 0;

    #[unroll(P::UNROLL)]
    while lane < count {
        // ...
        lane += 1;
    }
}
```

일반 정책 표현식은 `#![feature(generic_const_exprs)]`를 필요로 한다. `launch_contract` 필드, 클러스터 차원, 그리고 동적 공유 메모리 크기는 현재 문자 그대로 남아있다.

독립적인 PTX 전문화와 정책별 준비된 발사를 생성하는 두 가지 구체적인 정책에 대한 `policy_config`]](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/policy_config) 예시를 참조하십시오.

---

## 스레드 식별

```rust
use cuda_device::thread;

let idx     = thread::index_1d();                            // ThreadIndex<'_, Index1D>
let idx2d   = thread::index_2d::<128>();                     // Option<ThreadIndex<'_, Index2D<128>>>
let idx2d_r = thread::index_2d_runtime(&out);                // Option<ThreadIndex<'_, Runtime2DIndex>>
//            `out` is the DisjointSlice<T, Runtime2DIndex> the index will address;
//            its row width was bound once by the host via cuda_host::RowWidth.
let idx32   = thread::index_1d_u32(launch_context);          // ThreadIndex32<'_>
let pos32   = thread::coord_2d_u32(launch_context);          // ThreadCoord2D32<'_>

let tid_x  = thread::threadIdx_x();    // u32
let bid_x  = thread::blockIdx_x();     // u32
let bdim_x = thread::blockDim_x();     // u32
```

|기능                                    | 반환                                          | 설명 |
|:--------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------|
|`thread::index_1d()`                        | `ThreadIndex<'_, Index1D>`                       | 고유 선형 지수 (1D 그리드) |
|`thread::index_2d::<S>()`                   | `Option<ThreadIndex<'_, Index2D<S>>>`            | Const-stride 2D 인덱스; 일치하지 않는 스텝은 타입 오류입니다 |
|`thread::index_2d_runtime(&slice)`          | `Option<ThreadIndex<'_, Runtime2DIndex>>`        | 실행 시간 너비 2D 인덱스; 슬라이스에서 읽은 줄 너비 |
|`thread::index_1d_u32(launch_context)`      | `ThreadIndex32<'_>`                              | 1D 인덱스`u32`확인해야 합니다`u32`좌표 |
|`thread::coord_2d_u32(launch_context)`      | `ThreadCoord2D32<'_>`                            | 2차원 행/줄기: `u32`; 확인된 `u32` 좌표를 필요로 한다 |
|`thread::index_2d_row()`                    | `usize`                                          | 2D 행 지표 |
|`thread::index_2d_col()`                    | `usize`                                          | 2D 열 지표 |
|`thread::threadIdx_{x,y,z}()`               | `u32`                                            | 블록 안의 스레드 인덱스 |
|`thread::blockIdx_{x,y,z}()`                | `u32`                                            | 격자 안의 블록 인덱스 |
|`thread::blockDim_{x,y,z}()`                | `u32`                                            | 블록 크기 |

`thread::index_2d::<S>()`그리고 `thread::index_2d_runtime(&slice)`반환`None`계산된 열이 발걸음을 초과하면, 정렬되지 않은 2D 커널에서 오른쪽 가장자리의 꼬리를 건너뛰는 것입니다.

`index_2d::<S>`안전 const-stride 형태입니다; const generic는 증인 타입의 걸음을 코드화하므로 스레드는 다른 걸음을 사용할 수 없습니다. `index_1d`Y/Z 차원이 비활성이고 2D 인덱스는 비활동 Z가 필요합니다.`PreparedLaunch<K>`장치 검사 없이 증명합니다. 그렇지 않으면 장치가 잘못된 순위를 거부합니다: `index_1d`유효하지 않은 증인을 만들고 2D 보조자가 돌아옵니다`None`. 원시 발사에는 안전성이 없습니다. 왜냐하면 다른 메모리와 발사 의무가 확인되지 않았기 때문입니다. `index_2d_runtime`실행시에만 발걸음이 알려진 발사들을 포함합니다. 줄 너비는 슬라이스 안에 이동하고 호스트가 발사 패킷에 한 번 기록합니다 (`cuda_host::RowWidth`) 가 없으니`unsafe`그리고 단 한 번의 통화를 통해 의견이 갈라지는 것은 없습니다.`(row, col)`좌표와 주소 슬라이스는 자신의 줄 너비에 대해 그들을 해결합니다.[안전 모델](../gpu-safety/the-safety-model.md).

---

## 안전 평행 글쓰기  디조인트 슬라이스

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

|방법                  | 서명                                        | 설명 |
|:------------------------|:-------------------------------------------------|:---------------------------------------------------------------------|
|`get_mut_indexed`       | `() -> Option<(&mut T, ThreadIndex<'_, IS>)>`    | 한 번의 전화 양식: 증인으로 만들어서 해결합니다. |
|`get_mut`               | `(ThreadIndex<'_, IS>) -> Option<&mut T>`        | 명백한 목격자로부터의 제한된 변경 가능한 접근 |
|`get_unchecked_mut`     | `(usize) -> &mut T`                              | 안전하지 않고 통제되지 않은 접근 |
|`len`                   | `() -> usize`                                    | 요소 수 |

`get_mut_indexed`에 게이트됩니다`IndexSpace: IndexFormula`(에 의해 `Index1D`그리고 `Index2D<S>`에 대해`Runtime2DIndex`을 사용해서`thread::index_2d_runtime(&slice)`+ `get_mut(idx)`쌍; 슬라이스는 자신의 호스트-결정 라인 너비에 대한 목격자의 좌표를 해결 (`slice.row_width()`

고정 크기의 타일용은 `DisjointSlice<T, LinearTiles<N>>::tile_thread32`또는`DisjointSlice<T, RowMajorTiles<R, C, S>>::tile_2d32`각 방법은 한 번 완전한 타일을 확인하고,`at_const`다른 실행 시간 경계 확인 없이 알려진 위치에 액세스합니다. `S`호출자가 선언한 논리 행 너비이며 버퍼 레이아웃과 일치해야 합니다.{ref}`Check a tile once <check-a-tile-once>`.

---

## 공유 메모리

```rust
use cuda_device::{DynamicSharedArray, SharedArray, thread};

#[kernel]
pub fn tiled(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    unsafe { TILE[tid] = data[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... read from TILE ...
}

#[kernel]
pub fn dynamic(data: &[f32]) {
    static mut BUF: DynamicSharedArray<f32> = DynamicSharedArray::UNINIT;
    // Size set at launch via LaunchConfig::shared_mem_bytes
}
```

 타입                      | 설명                                               
|:--------------------------|:----------------------------------------------------------|
 `SharedArray<T, N>`       | 컴파일 시간 크기의 블록 스코프 공유 메모리            
 `SharedArray<T, N, 128>`  | 128바이트 정렬 (TMA 목적지) 에 필요한)   
 `DynamicSharedArray<T>`   | 실행 시간 크기의 공유 메모리 (`LaunchConfig`를 통해 설정)      
 `SharedPtr<T>`            | 이 블록의 공유 메모리에 포인터를 `addrspace(3)`종류별로 

둘 다`!Sync` 동시 접속은 명시적인 장벽을 요구합니다.

---

## 동기화

### 블록 수준

```rust
thread::sync_threads();   // __syncthreads() equivalent
```

### 관리된 장애물 (호퍼+)

```rust
use cuda_device::{ManagedBarrier, Ready, TmaBarrierHandle, Uninit};

// Typestate lifecycle: Uninit → Ready → Invalidated
let bar: TmaBarrierHandle<Uninit> = TmaBarrierHandle::from_static(ptr);
let bar: TmaBarrierHandle<Ready> = unsafe { bar.init(thread_count) };
let token = bar.arrive();
bar.wait(token);
unsafe { bar.inval() };
```

 작전                   | 설명                                      
|:----------------------------|:-------------------------------------------------|
 `.init(count)`              | 예상 도착 수와 함께 장벽을 초기화   
 `.arrive()`                 | 신호 도착, 반환`BarrierToken`           
 `.arrive_expect_tx(bytes)`  | 도착하고 예상 TX 바이트 수를 설정 (TMA를 위해)  
 `.wait(token)`              | 모든 도착 + TX 완료 될 때까지 차단           
 `.inval()`                  | 무효 장벽 (정화)                     

---

## 워프 원시자

```rust
use cuda_device::warp;

let lane = warp::lane_id();      // 0–31
let wid  = warp::warp_id();

// Shuffle
let partner = warp::shuffle_xor_f32(val, mask);
let from_above = warp::shuffle_down_f32(val, delta);
let from_below = warp::shuffle_up_f32(val, delta);
let from_lane  = warp::shuffle_f32(val, src_lane);

// u32 is the unsuffixed form
let partner_u = warp::shuffle_xor(val_u32, mask);

// Vote
let all_true = warp::all(predicate);
let any_true = warp::any(predicate);
let mask     = warp::ballot(predicate);
let count    = warp::popc(predicate); // == ballot(predicate).count_ones()
```

### 셔플 작업

부가적인 이름은 입니다.`u32``_f32`, `_f64`그리고 `_u64`다른 너비도 있습니다. 각 너비에는 이 있습니다.`_sync` 은 이 없습니다.`_i32`변수  재해석`i32`그리고 을 사용하세요`u32` 은 비트들을 움직이고 해석하지 않기 때문에

 기능                                    | 설명                       
|:--------------------------------------------|:----------------------------------|
 `shuffle_xor(val, mask)` + `_f32/_f64/_u64` | 차선으로 교환`id ^ mask`    
 `shuffle_down(val, delta)` + `_f32/_f64/_u64` | `id + delta`     
 `shuffle_up(val, delta)` + `_f32/_f64/_u64` | `id - delta`       
 `shuffle(val, src)` + `_f32/_f64/_u64`      | 특정 차선에서 읽습니다           

### 투표 운영

|기능       | 반환  | 설명 |
|:---------------|:---------|:---------------------------------------------|
|`all(pred)`    | `bool`   | 만약 전조가 모든 경로에 유효하다면 true |
|`any(pred)`    | `bool`   | 만약 전조가 모든 경로에 유효하다면 true |
|`ballot(pred)` | `u32`    | 전제가 사실인 경로의 비트 마스크 |
|`popc(pred)`   | `u32`    | 선명사가 사실인 랜의 수 |

---

## 원자력

### 스코프 GPU Atomics

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);

// In kernel:
COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
let old = COUNTER.load(AtomicOrdering::Acquire);
```

 적용 범위                                        | 종류                         
|:---------------------------------------------|:------------------------------|
 `DeviceAtomic{U32,I32,U64,I64,F16,F32,F64}`  | `.gpu`적용 범위                  
 `BlockAtomic{U32,I32,U64,I64,F16,F32,F64}`   | `.cta`적용 범위                  
 `SystemAtomic{U32,I32,U64,I64,F16,F32,F64}`  | `.sys` 범위 (CPU-GPU 공유) 

21가지 유형: 세 가지 범위의 각 값의 폭은 7개입니다.`F16`변수들은 와 같은 표면을 차지합니다.`F32`/`F64`- `load`, `store`, `fetch_add`, `fetch_sub`, `swap`-- `fetch_add`/`fetch_sub`하드웨어로 내려가기`atom.add.noftz.f16`.

`core::sync::atomic`종류 (`AtomicU32`, `AtomicBool`, 등) 또한 GPU 코드로 컴파일하여 시스템 범위로 기본 설정됩니다.

nvptx64에서, `core::sync::atomic::AtomicPtr<T>`는 낮출을 통해 포인터 유형을 보존하면서 일반 주소 공간 포인터 값에 대한 `load`, `store`, #`swap`, 그리고 #`compare_exchange`를 지원합니다. 로컬 원자 저장소는 일반적인 스레드 개인 액세스를 사용합니다. 글로벌 및 공유 저장소는 보조자가 일반 포인터를 수신할 때 포함해서 원자 순서와 범위를 유지합니다. 레거시 NVVM은 포인터 교환 및 비교 교환을 위해 범위를 PTX를 사용하고 포인터 타입의 연산과 결과를 유지합니다

---

## TMA  텐서 메모리 가속기 (호퍼+)

```rust
use cuda_device::tma::TmaDescriptor;
use cuda_device::tma::{cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_g2s};

// Host: build descriptor (128 bytes, opaque)
// Device: issue async bulk copy
cp_async_bulk_tensor_2d_g2s(smem_ptr, &desc, coord_x, coord_y, barrier_ptr);
cp_async_bulk_commit_group();
```

 기능                                      | 설명                          
|:----------------------------------------------|:-------------------------------------|
 `cp_async_bulk_tensor_{1..5}d_g2s(...)`       | 글로벌 → 공유 아시크 대량 복사      
 `cp_async_bulk_tensor_{1..5}d_g2s_cta(...)`   | 글로벌 → 이 CTA의 `SharedPtr`      
 `cp_async_bulk_tensor_{1..5}d_s2g(...)`       | 공유 → 글로벌 비동시 대량 복사      
 `cp_async_bulk_tensor_2d_g2s_multicast(...)`  | 클러스터의 모든 CTA에 멀티캐스트     
 `cp_async_bulk_commit_group()`                | 출간된 복사본을 출품합니다            
 `cp_async_bulk_wait_group(n)`                 | ≤ n 그룹이 남아있을 때까지 기다립니다         

---

## 클러스터 프로그래밍 (호퍼+)

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();        // This block's rank in the cluster
let size = cluster::cluster_size();      // Number of blocks in cluster
cluster::cluster_sync();                 // Barrier across all cluster blocks

// Distributed Shared Memory. Both are `unsafe fn` and both take the *local*
// pointer plus the target rank; they do the rank mapping themselves, so
// never pass a pointer already returned by `map_shared_rank` back in.

// `map_shared_rank` returns a real pointer into the target block's shared
// memory (cluster-shared address space). Plain reads and writes through it
// compile to ld.shared::cluster / st.shared::cluster and just work on sm_90+.
let remote_ptr = unsafe { cluster::map_shared_rank(local_ptr, target_rank) };
let val = unsafe { *remote_ptr };  // ld.shared::cluster

// Fixed-width alternative: map and read one u32 in a single call.
let val = unsafe { cluster::dsmem_read_u32(local_u32_ptr, target_rank) };
```

---

## 텐서 코어  WGMMA (호퍼, SM 90)

```rust
use cuda_device::wgmma;

wgmma::wgmma_fence();
wgmma::wgmma_commit_group();
wgmma::wgmma_wait_group::<0>();
```

워프그룹 MMA: 4개의 워프 (128개의 스레드) 에이션 매트릭스 공유 메모리에서 멀티플리-아쿠뮬레이트. SMEM 디스크립터에 의해 설명된 오페란드; 레지스터에서 축적자.

---

## 텐서 코어  tcgen05 (블랙웰, SM 100+)

```rust
use cuda_device::SharedArray;
use cuda_device::tcgen05::{TmemGuard, TmemReady, TmemUninit};

static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

let guard = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
let guard = unsafe { guard.alloc() };   // TmemUninit → TmemReady
// ... issue MMA, read results via guard.address() ...
let _guard = unsafe { guard.dealloc() }; // TmemReady → TmemDeallocated
```

단일 스레드 MMA 발행 전용 텐서 메모리 (TMEM). `TmemGuard`는 타입 상태: `TmemUninit → TmemReady → TmemDeallocated`로 TMEM 수명을 관리합니다. N_COLS는 범위 [32, 512]의 2의 힘이어야합니다.

---

## 호스트 측: 커널 런칭

### 타이핑 동기화

```rust
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();
let module = kernels::load(&ctx).unwrap();

let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
let mut output = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

// SAFETY: this is a 1D launch and all buffers contain n elements.
unsafe {
    module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut output)
}
.unwrap();
```

### 타입: 아시크

```rust
use cuda_async::simt::device_operation::DeviceOperation;

let module = kernels::load_async(0)?;
// SAFETY: this is 1D, buffers contain n elements, and module/scheduler share a context.
let op = unsafe {
    module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut output)
}?;

op.sync()?;       // blocking
// or: op.await?;  // async with tokio
```

원시 생성 전화는 안전하지 않습니다.`LaunchConfig`커널에 묶여 있지 않습니다.`#[launch_contract]`대신 을 사용한다`LaunchConfig1D/2D/3D`확인된 을 만들 수 있습니다`PreparedLaunch<K>`, 그리고 안전하게 발사합니다. `cuda_launch!`그리고 `cuda_launch_async!`명확한 모듈 로딩과 사용자 지정 런칭 코드를 위한 낮은 수준의 API가 안전하지 않습니다.

### 랜치 컨피그

 방법                                                   | 설명                                  
|:---------------------------------------------------------|:---------------------------------------------|
 `LaunchConfig::for_num_elems(n)`                         | `n` 요소에 대한 자동 구성 그리드/블록   
 `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }` | 직접 구조 건설                   

---

## 호스트 측: 가상 메모리 관리

### VMM의 생명주기

 API                                         | 목적                                     
|:--------------------------------------------|:--------------------------------------------|
 `vmm::allocation_granularity(device)`       | 필요한 할당 분량 을 검색   
 `vmm::align_size(size, granularity)`        | 필요한 곡물량에 대한 크기를 둥글게    
 `PhysicalAllocation::new(device, size)`     | 물리적 메모리를 할당합니다                    
 `VirtualReservation::new(size, alignment)`  | 가상 주소 범위를 예약하세요             
 `Mapping::new(va, size, &physical, offset)` | VA 범위로 물리적 메모리를 매핑합니다         
 `vmm::set_access(va, size, devices)`        | 선택된 장치에 읽기/쓰기 액세스를 부여합니다 

가상 예약과 물리적 할당 전에 지도를 삭제해야합니다.

---

## 호스트: 피어 액세스

 API                                   | 목적                                           
|:--------------------------------------|:--------------------------------------------------|
 `peer::can_access_peer(from, to)`     | 토폴로지가 직접 접근을 지원하는지 여부를 검색합니다 
 `peer::enable_peer_access(from, to)`  | 일방적인 피어 액세스 허용                
 `peer::disable_peer_access(from, to)` | 일방적인 피어 액세스를 비활성화합니다               

피어 액세스 방향입니다. 두 장치가 액세스 시작해야 할 때 두 방향 모두 활성화합니다.

---

## 호스트: 커널 계열

 타입 | 목적 
|:-----|:--------|
 `KernelFamily<Id, Entry, Meta, N>` | 미리 작성된 변수들의 고정된 집합 
 `KernelVariant<Id, Entry, Meta>` | 안정적인 ID, 호출 가능한 입력, 정책 메타데이터 
 `KernelProblem<Variant>` | 변수가 지원 여부를 확인합니다 
 `KernelSelector<Problem, Variant, Id>` | 이미 지원 가능한 변종 중 하나를 선택합니다 
 `KernelSelectionCache<Problem, Id>` | 안정적인 선택 ID를 저장합니다 
 `NoKernelSelectionCache` | 캐시 기능을 비활성화합니다 
 `SelectionMode::Auto` | 검증된 캐시 결과를 사용하거나 선택자를 호출합니다 
 `SelectionMode::Force(id)` | 캐시와 선택자를 우회하지만 여전히 적정성을 확인합니다 
 `SelectedVariant` | 선택된 변수와 출처를 반환합니다 
 `SelectionSource` | 보고가 오버라이드, 캐시 또는 선택자의 출처 

`KernelFamily::try_new`는 빈 가족, 빈 가족 이름, 복제된 변수 ID를 거부합니다. 가족 이름과 개정이 캐시 네임스페이스를 형성합니다. 변수 의미, 회원, 순서 또는 선택 정책 변경될 때마다 개정을 증가시킵니다.

전체 선택 모델과 예제를 위해 [Kernel Families]](../gpu-programming/kernel-families.md)를 참조하십시오.

---

## 디버그 기능

```rust
use cuda_device::debug;

let t = debug::clock64();       // Cycle counter
debug::trap();                  // Abort kernel
debug::breakpoint();            // cuda-gdb breakpoint
unsafe { cuda_device::barrier::nanosleep(1000) }; // Sleep ~1μs
debug::prof_trigger::<7>();     // Nsight profiler trigger
```

---

## 빠른 참조 표

### cuda-device 모듈

|모듈               | 설명                                                      | 미네 SM |
|:---------------------|:-----------------------------------------------------------------|:---------|
|`thread`             | 스레드/블록 ID, `index_1d`, `sync_threads`                     | 모두 |
|`config`             | 컴파일 타임 정책, 형상, 타일, 원자, 레이아웃, 메모리 공간, 스코프 | 모두 |
|`disjoint`           | `DisjointSlice<T>` 발사 증명서를 작성한 글    | 모두 |
|`shared`             | `SharedArray<T, N>`, `DynamicSharedArray<T>`, #`SharedPtr<T>`     | 모두 |
|`warp`               | 셔플, 투표, 일치, 차선/워프 ID                               | 모두 |
|`atomic`             | 범주 원자 (장치/블록/시스템)                             | sm_70+ |
|`debug`              | `clock64`, `trap`, `breakpoint`, `gpu_printf!`                   | 모두 |
|`fence`              | `threadfence_block` / `threadfence` / #`threadfence_system`       | 모두 |
|`grid`               | 그리드 스코프 된 `sync` (협동 커널 실행)                 | sm_70+ |
|`cooperative_groups` | 타이핑 된 손잡이, 워프/블록 감소 및 스캔                   | 모두 |
|`barrier`            | `ManagedBarrier` TMA/MMA를 위한 비동기 장애물                    | sm_90+ |
|`cluster`            | 스레드 블록 클러스터, DSMEM                                     | sm_90+ |
|`tma`                | `TmaDescriptor`, 대용량 텐서 복사 (1D5D)                      | sm_90+ |
|`wgmma`              | 워프그룹 MMA (비장/공약/기차)                                | sm_90 |
|`tcgen05`            | 5세대 텐서 코어, TMEM, `TmemGuard`                          | sm_100+ |
|`cusimd`             | `CuSimd<T, N>`, `Float2`/ `Float4`                                | 모두 |
|`clc`                | 클러스터 발사 제어                                           | sm_100+ |

### 크레이트 지도

 크레이트             | 역할                                                                   
|:------------------|:-----------------------------------------------------------------------|
 `cuda-device`     | 장치의 본질 및 유형 (`#![no_std]`)                             
 `cuda-macros`     | Proc 매크로 (`#[kernel]`, `#[device]`, #`gpu_printf!`, @`ptx_asm!`)      
 `cuda-host`       | 타입 모듈 로딩 더 낮은 레벨의 발사 보조자                     
 `cuda-core`       | 컨텍스트, 스트림, 버퍼, VMM 및 P2P를 위한 안전한 RAII 래퍼        
 `cuda-async`      | `DeviceOperation`, `DeviceFuture`, #`DeviceBox<T>`                      
 `cuda-bindings`   | 생화질`bindgen`FFI에서 `cuda.h`                                          
 `cargo-oxide`     | 화물 하위 명령 (`cargo oxide run`, `build`, `inspect`, `clean`, `sanitize`, `debug` 

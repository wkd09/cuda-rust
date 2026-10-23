# 용어집

각 용어는 cuda-oxide 프로젝트와 이 책에서 사용되는 의미로 정의합니다.

---

## ABI 스칼라화 (ABI Scalarization)

복합 타입을 호스트와 장치가 함께 이해할 수 있는 형태로 분해하는 과정입니다. 슬라이스는 모든 호출 경계에서 `(ptr, len)` 쌍으로 평탄화되며 함수 내부에서 `insertvalue` / `extractvalue`를 통해 재구성됩니다. 값으로 전달되는 구조체와 클로저도 장치 함수 간 호출에서는 같은 방식으로 평탄화됩니다. 다만 **커널** 경계에서는 호스트 실행기가 전달하는 단일 슬롯 패킷과 일치하도록 하나의 byval `.param`으로 전달됩니다. 이 방식은 장치 선언은 평탄화됐지만 호스트 패킷은 단일 슬롯인 불일치를 방지합니다.

## 블록 (스레드 블록)

같은 스트리밍 멀티프로세서에서 실행되며 공유 메모리에 접근할 수 있는 스레드 그룹입니다. 블록 안의 스레드는 `sync_threads()`로 동기화할 수 있습니다. `blockIdx`로 위치를 식별하고 `blockDim`으로 크기를 나타냅니다.

## 클러스터 (스레드 블록 클러스터)

Hopper 이상(sm_90)의 기능으로, 최대 16개 스레드 블록을 같은 GPC에서 함께 스케줄링하도록 보장합니다. 클러스터의 블록들은 분산 공유 메모리(DSMEM)로 서로의 공유 메모리에 접근하고 `cluster_sync()`로 동기화할 수 있습니다. `#[cluster_launch]`로 설정합니다.

## 코드 생성 백엔드 (Codegen Backend)

`rustc_codegen_cuda` 크레이트를 가리킵니다. dylib로 로드되는 사용자 정의 rustc 백엔드로, 컴파일 중 MIR을 가로채 `dialect-mir` → `mem2reg` → 주석 기반 루프 언롤 → LLVM dialect → LLVM IR → PTX 순서로 낮춥니다. 일반 호스트 바이너리와 함께 PTX를 출력합니다.

## `cuda-async`

저장소 루트의 `cuda-async/`에서 관리하며 cutile-rs와 공유하는 비동기 실행 계층입니다. cuda-oxide 모델은 `cuda_async::simt` 아래에 있습니다. `DeviceOperation`은 지연 실행 GPU 작업 설명, `DeviceFuture`는 스트림에 바인딩된 실행, `DeviceBox<T>`는 장치 소유 메모리를 나타냅니다. `zip!`, `and_then`, `value()`로 작업을 조합합니다.

## `cuda-device`

모든 GPU intrinsic과 타입을 제공하는 `#![no_std]` 장치 측 크레이트입니다. 스레드 식별, 공유 메모리, 워프 연산, 배리어, TMA, 텐서 코어, 원자 연산, 디버그 기능을 포함합니다.

## `cuda-core`

CUDA Driver API를 감싸는 안전한 RAII 래퍼입니다. `CudaContext`, `CudaStream`, `DeviceBuffer<T>`, 모듈 로딩을 제공하고 호스트 측 GPU 컨텍스트 및 메모리 관리를 담당합니다. 저장소 루트의 `cuda-core/`에서 관리하며 cutile-rs와 공유합니다. `simt::LaunchConfig`, `simt::vmm`, `simt::peer`, `simt::memory`를 제외한 cuda-oxide 인터페이스는 크레이트 루트에서 다시 내보냅니다.

## `DeviceOperation`

할당, 커널 실행, 데이터 전송 등 GPU 작업을 지연 실행 가능한 형태로 표현합니다. `.sync()` 또는 `.await`를 호출하기 전에는 실행되지 않습니다. `zip!`으로 병렬 조합하거나 `and_then`으로 순차 조합할 수 있습니다.

## `DisjointSlice<T, IndexSpace>`

커널의 안전한 가변 출력 추상화입니다. 자체 타입 매개변수와 `IndexSpace`가 일치하는 `ThreadIndex`만 받으며, 경계를 검사해 `Option<&mut T>`를 반환합니다. 준비된 실행 구성의 인덱스 공간이 타입과 일치하면 각 스레드는 자신의 원소에만 쓸 수 있습니다. `get_mut_indexed()`는 증인을 만들고 가변 참조까지 한 번에 얻는 간편 메서드입니다.

## 분산 공유 메모리 (Distributed Shared Memory, DSMEM)

Hopper 이상에서 클러스터 내부 블록이 전역 메모리를 거치지 않고 서로의 공유 메모리를 직접 읽고 쓸 수 있게 합니다. `cluster::map_shared_rank()`로 로컬 공유 메모리 포인터를 원격 주소로 변환합니다.

## 그리드 (Grid)

커널 실행에서 스레드 블록을 구성하는 최상위 단위입니다. `gridDim`으로 크기를 나타내는 3차원 스레드 블록 배열입니다. 전체 스레드 수는 `gridDim × blockDim`입니다.

## HMM (이기종 메모리 관리)

Linux 커널 6.1.24 이상 기능으로, 명시적인 `cudaMemcpy` 없이 페이지 폴트를 통해 GPU가 호스트 메모리에 직접 접근할 수 있습니다. cuda-oxide는 클로저의 참조 캡처에 HMM을 사용하므로 GPU가 호스트 주소를 투명하게 읽습니다. UMM(Unified Memory Management)이라고도 합니다.

## 레인 (Lane)

워프 안의 개별 스레드입니다. 레인 번호는 0–31이며 `warp::lane_id()`로 확인합니다. 워프 셔플 및 투표 연산은 공유 메모리나 배리어 없이 레인 사이에서 데이터를 교환합니다.

## LTOIR (링크 시간 최적화 IR)

장치 측 링크 시간 최적화에 사용하는 중간 표현입니다. cuda-oxide는 libNVVM용 NVVM IR을 출력하고, 이를 LTOIR로 컴파일해 `nvJitLink`로 Rust 장치 코드와 CUDA C++ 장치 코드를 링크할 수 있습니다.

## `ManagedBarrier<State, Kind, ID>`

Hopper 이상에서 비동기 작업을 위한 타입 상태 배리어입니다. 컴파일 시점에 수명 주기(`Uninit → Ready → Invalidated`)와 용도(`TmaBarrier`, `MmaBarrier`, `GeneralBarrier`)를 추적합니다. 잘못된 상태 전이는 컴파일 오류가 됩니다.

## 모노모피제이션 (Monomorphization)

Rust 컴파일러가 제네릭 함수에서 사용된 구체 타입마다 특수화된 복사본을 만드는 과정입니다. cuda-oxide는 장치 코드의 모노모피제이션을 완전히 지원하므로 `scale::<f32>`와 `scale::<f64>`는 각각 별도의 PTX 함수가 됩니다.

## `PreparedLaunch<K>`

특정 커널 `K`에 대해 실행 구성을 검사했음을 나타내는 호스트 측 재사용 증거입니다. 생성된 안전 메서드는 이 브랜드가 붙은 값을 받습니다. 원시 `LaunchConfig`를 생성 메서드에 직접 전달하면 커널의 인덱싱이나 리소스 요구사항이 증명되지 않으므로 unsafe 호출이 됩니다.

## Pliron

cuda-oxide 컴파일 파이프라인의 중간 표현으로 사용하는 Rust 기반 MLIR 영감 IR 프레임워크입니다. MIR을 Pliron IR로 가져오고 dialect 패스를 적용한 뒤 LLVM IR로 내보냅니다.

## PTX (병렬 스레드 실행)

NVIDIA GPU 커널을 위한 저수준 가상 ISA입니다. cuda-oxide 컴파일러는 PTX를 주요 출력으로 생성하며, CUDA 드라이버는 모듈 로드 시 이를 네이티브 GPU 기계어(SASS)로 JIT 컴파일합니다.

## `SharedArray<T, N, ALIGN>`

컴파일 시 크기가 정해지는 블록 범위 공유 메모리 배열입니다. 커널 함수에서 `static mut`으로 선언합니다. 선택적 정렬 매개변수를 제공하며 TMA 대상에는 `ALIGN=128`을 사용할 수 있습니다. 공유 메모리는 `!Sync`이므로 접근에는 `unsafe`가 필요합니다.

## SM (스트리밍 멀티프로세서)

NVIDIA GPU의 기본 처리 장치입니다. 각 SM에는 자체 레지스터, 공유 메모리, 워프 스케줄러와 텐서 코어를 포함한 실행 파이프라인이 있습니다. 하드웨어 스케줄러가 스레드 블록을 SM에 배정합니다.

## `sync_threads()`

블록 수준 배리어입니다. 스레드 블록의 모든 스레드가 이 지점에 도달해야 각 스레드가 다음 단계로 진행합니다. CUDA C++의 `__syncthreads()`와 동등하며 `llvm.nvvm.barrier0()`로 낮춰집니다.

## 텐서 코어 (Tensor Cores)

행렬 곱 누산 전용 하드웨어 장치입니다. WGMMA(Hopper, sm_90)는 공유 메모리에서 워프그룹 단위로 작동합니다. tcgen05(Blackwell, sm_100 이상)는 전용 텐서 메모리(TMEM)를 사용해 단일 스레드가 명령을 발행합니다.

## `ThreadIndex<'kernel, IndexSpace>`

신뢰된 인덱스 함수만 만들 수 있는 불투명한 증인입니다. 세 가지 형태가 있습니다.

- `thread::index_1d() -> ThreadIndex<'_, Index1D>`는 항상 증인을 반환합니다. 블록과 그리드의 Y/Z 차원이 1일 때만 고유하며 `domain = 1`인 준비된 실행이 이를 증명합니다.
- `thread::index_2d::<S>() -> Option<ThreadIndex<'_, Index2D<S>>>`의 행 간격은 const generic입니다. 따라서 `DisjointSlice<T, Index2D<S>>`는 동일한 `S`의 증인만 받으며 서로 다른 간격을 섞으면 타입 오류가 발생합니다.
- `thread::index_2d_runtime(&slice) -> Option<ThreadIndex<'_, Runtime2DIndex>>`는 실행 시점에만 알려지는 간격에 사용하며 안전합니다. 행 너비는 호스트가 실행 패킷에 한 번 기록합니다. 증인은 스레드의 `(row, col)` 좌표를 보관하고 대상 슬라이스는 자체 너비로 좌표를 해석하므로 해당 슬라이스를 인덱싱하는 모든 스레드의 행 너비가 일치합니다.

증인은 `!Send + !Sync + !Copy + !Clone`이며 `'kernel` 수명에 묶입니다. 따라서 스레드가 공유 메모리를 통해 증인을 우회 전달할 수 없고 커널 본문보다 오래 유지할 수도 없습니다.

## TMA (텐서 메모리 가속기)

Hopper 이상에서 전역 메모리와 공유 메모리 사이의 대량 비동기 복사를 수행하는 하드웨어 장치입니다. 호스트에서 만드는 128바이트 불투명 설명자 `TmaDescriptor`와 `cp_async_bulk_tensor_*` intrinsic을 통해 작동합니다. 완료 여부는 `ManagedBarrier`로 추적합니다.

## `TmemGuard<State, N_COLS>`

tcgen05 MMA 연산의 누산기 저장소인 Blackwell 텐서 메모리(TMEM)를 감싸는 타입 상태 래퍼입니다. TMEM 수명(`TmemUninit → TmemReady → TmemDeallocated`)을 관리하며 잘못된 전이는 컴파일 오류가 됩니다.

## 워프 (Warp)

SM에서 명령을 동시에 실행하는 32개 스레드 그룹입니다. 워프는 하드웨어의 최소 스케줄링 단위입니다. 셔플과 투표 등 워프 연산은 공유 메모리나 동기화 배리어 없이 약 한 사이클 만에 레인 간 데이터를 교환합니다.

## 워프그룹 (Warpgroup)

Hopper에서 WGMMA 연산을 함께 수행하는 연속된 네 개의 워프(128개 스레드)입니다. 워프그룹 단위 행렬 곱 누산 명령을 발행하는 단위입니다.

## WGMMA (워프그룹 행렬 곱 누산)

Hopper(sm_90) 텐서 코어 명령어 집합입니다. 네 워프가 함께 공유 메모리 피연산자에서 MMA를 발행하고 레지스터 누산기에 결과를 기록합니다. `ManagedBarrier`를 이용해 커밋하고 대기하는 비동기 실행 방식입니다.

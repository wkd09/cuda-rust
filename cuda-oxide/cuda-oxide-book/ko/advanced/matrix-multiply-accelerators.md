# 행렬 곱 가속기

최신 NVIDIA GPU에는 행렬 곱 누산(MMA) 연산 전용 하드웨어가 있으며, 일반적으로 **텐서 코어**라고 부릅니다. 텐서 코어는 표준 부동소수점 ALU보다 훨씬 높은 처리량으로 작은 행렬 타일에 `D = A × B + C`를 계산합니다. Hopper H100은 FP16 MMA에서 1000 TFLOPS 이상을 내며, 같은 칩의 스칼라 FP16 처리량은 약 200 TFLOPS입니다. 딥러닝, HPC, 신호 처리 등 행렬 곱이 포함된 대부분의 작업에서 성능을 좌우하는 곳은 텐서 코어입니다.

cuda-oxide는 두 세대의 행렬 가속기를 제공합니다. Hopper(SM 90)의 **WGMMA**와 Blackwell(SM 100)의 **tcgen05**입니다. 이 장에서는 두 프로그래밍 모델과 앞선 장의 TMA 및 배리어 기능과의 통합을 다룹니다.

:::{seealso}
WGMMA의 하드웨어 사양, 연산 모양, 원소 타입, 동기화 요구사항은 [CUDA 프로그래밍 가이드 — 워프그룹 수준 행렬 연산](https://docs.nvidia.com/cuda/cuda-programming-guide/#warpgroup-level-matrix-operations)을 참조하세요.
:::

---

## 전체 구조```{figure} images/tensor-core-datapath.svg
:align: center
:width: 100%

Hopper WGMMA와 Blackwell tcgen05의 데이터 경로입니다. 두 방식 모두 설명자를 통해 공유 메모리에서 피연산자를 읽습니다. WGMMA는 워프그룹 집합 방식으로 스레드별 레지스터에 누적합니다. tcgen05는 단일 스레드가 발행하고 전용 텐서 메모리(TMEM)에 누적합니다.
```세대가 바뀌면서 피연산자는 연산 장치에 더 가까이 이동하고, 명령을 발행하는 범위는 넓어지며, 프로그래머는 더 큰 타일에 더 적은 명령을 작성하게 됩니다. cuda-oxide는 하나의 만능 추상화 대신 세대별 API로 이러한 변화를 반영합니다.

---

## WGMMA — Hopper (SM 90)

WGMMA(**W**arp**G**roup **M**atrix **M**ultiply-**A**ccumulate)는 **워프그룹 집합 연산**입니다. 워프 4개(스레드 128개)가 협력해 행렬 타일을 계산합니다. A와 B 피연산자는 SMEM 설명자를 통해 공유 메모리에서 읽고 결과는 스레드별 레지스터에 누적합니다.

### 프로그래밍 모델

1. **TMA 로드**가 A와 B 타일을 공유 메모리로 가져옵니다([텐서 메모리 가속기](tensor-memory-accelerator.md) 참조).
2. **SMEM 설명자**가 각 피연산자의 베이스 주소, stride, swizzle 모드를 인코딩합니다.
3. **WGMMA 명령**이 설명자를 사용해 누산기를 갱신합니다. 비동기 명령이며 배리어에 커밋됩니다.
4. **배리어 대기**로 MMA 완료를 확인한 뒤 누산기를 읽습니다.

### 하드웨어 지원 연산 모양

WGMMA는 항상 M=64(행)이고 N과 K는 원소 타입에 따라 다릅니다.

| 원소 타입 | K | N 옵션 |
|:---|---:|:---|
| f16, bf16 | 16 | 64, 128, 256 |
| tf32 | 8 | 64, 128, 256 |

명령 하나가 64×N×K 타일을 계산합니다. K 차원이 더 크면 WGMMA 명령을 루프에서 여러 번 호출해 같은 레지스터 타일에 누적합니다.

### cuda-oxide API 개요

cuda-oxide는 하드웨어 명령에 직접 대응하는 저수준 intrinsic으로 WGMMA를 제공합니다. 일반적인 사용 패턴은 다음과 같습니다.```rust
use cuda_device::wgmma::{
    make_smem_desc, wgmma_commit_group, wgmma_fence, wgmma_mma_m64n64k16_f32_bf16, wgmma_wait_group,
};

// After TMA has loaded A and B tiles into shared memory...

// Build SMEM descriptors for the loaded tiles
let a_desc = unsafe { make_smem_desc(tile_a_ptr as *const u8) };
let b_desc = unsafe { make_smem_desc(tile_b_ptr as *const u8) };

// Accumulator (32 floats per thread = 64×64 tile across the warpgroup)
let mut acc = [[0.0f32; 8]; 4];

unsafe {
    wgmma_fence();
    wgmma_mma_m64n64k16_f32_bf16(&mut acc, a_desc, b_desc);
    wgmma_commit_group();
    wgmma_wait_group::<0>();
}

// `acc` is now safe to observe.
```BF16 m64n128 전체 대기 변형도 같은 동기화 규약을 사용하지만 스레드마다 누산기 값 64개를 사용합니다.```rust
use cuda_device::wgmma::wgmma_mma_m64n128k16_f32_bf16;

let mut wide_acc = [[0.0f32; 8]; 8];
unsafe {
    wgmma_fence();
    wgmma_mma_m64n128k16_f32_bf16(&mut wide_acc, a_desc, b_desc);
    wgmma_commit_group();
    wgmma_wait_group::<0>();
}
```TF32에는 `wgmma_mma_m64n64k8_f32_tf32`를 사용합니다. Hopper의 TF32 WGMMA 하드웨어 연산 모양은 K=8이므로 K=16 TF32 호환 진입점은 지원되지 않습니다.

### 현재 cuda-oxide의 WGMMA 낮추기

컴파일러는 `m64n64k16.f32.bf16.bf16`을 보수적인 낮추기 모양 세 가지로 지원합니다. `m64n64k16.f32.f16.f16`은 정형화된 선형 전체 대기와 계수 K 루프 영역에서, `m64n64k8.f32.tf32.tf32`는 정형화된 선형 전체 대기 영역에서, `m64n128k16.f32.bf16.bf16`은 정형화된 선형 전체 대기 영역에서 지원합니다. 허용된 모든 경우 비동기 누산기 수명 전체가 수렴형 인라인 PTX 구문 하나 안에 유지됩니다. 따라서 WGMMA 그룹이 대기 중일 때 LLVM이 spill 경계를 삽입할 수 없습니다.

**m64n64 선형 전체 대기.** 정형화된 `[[f32; 8]; 4]` 누산기는 WGMMA 영역 전에 SSA `f32` 값 32개로 읽고 마지막 `wait_group<0>` 뒤에 한 번 저장합니다. BF16, F16, TF32는 이 값 기반 운반 방식을 사용합니다. 지원하지 않는 BF16 전체 대기 포인터 모양은 기존 지연 포인터 방식 낮추기를 유지합니다. F16과 TF32에는 포인터 방식 대체 경로가 없습니다.

**m64n128 BF16 선형 전체 대기.** 정형화된 `[[f32; 8]; 8]` 누산기는 동종 `m64n128k16.f32.bf16.bf16` 명령 하나 이상, 커밋 한 번, 마지막 `wait_group<0>`을 거치는 SSA `f32` 값 64개로 유지됩니다. 누산기 원소 주소는 융합된 인라인 PTX 범위 이후에 다시 계산하며 범위 안에 계속 살려 두지 않습니다. 이 모양은 포인터 방식 대체 경로, 계수 루프 낮추기, 부분 대기 파이프라인 낮추기를 지원하지 않습니다.

**정형화된 계수 K 루프.** 반복마다 MMA를 하나 실행하는 컴파일 시점 계수 BF16 또는 F16 m64n64 루프와 affine `u64` 설명자 재귀식에 대해 cuda-oxide는 루프 제어와 설명자 산술을 WGMMA 명령과 같은 수렴형 PTX 영역으로 이동합니다. 따라서 누산기는 K 루프 전에 한 번 읽고 루프 반복마다 왕복하는 대신 마지막 대기 뒤 한 번 저장합니다.

**정적 부분 대기 파이프라인.** BF16 m64n64의 `wait_group<N>`에서 `N > 0`이면 서로 독립된 누산기 슬롯 `N + 1`개를 사용합니다. 그룹은 각각 커밋하며 가장 오래된 슬롯이 부분 대기로 안전해진 뒤에만 라운드 로빈으로 슬롯을 재사용합니다. 예를 들어 그룹 두 개를 다음처럼 겹쳐 실행할 수 있습니다.```text
wgmma_fence
mma acc0
commit_group
mma acc1
commit_group
wait_group<1>
wait_group<0>
```긴 파이프라인은 누산기를 재사용하기 전에 슬롯 일정을 반복하고 부분 대기합니다. 누산기 값이 융합 영역을 벗어나기 전 마지막 `wait_group<0>`은 필수입니다.

선택은 의도적으로 fail-closed 방식입니다. 동적 부분 대기, 지원하지 않는 제어 흐름, 잘못된 누산기 일정, F16 부분 대기 모양, TF32 계수 루프 및 부분 대기 모양, m64n128 계수 루프 및 부분 대기 모양, 정형화되지 않은 F16/TF32/m64n128 누산기, 이전 K=16 TF32 호환 진입점은 비행 중인 누산기를 LLVM에 노출하는 대신 거부됩니다.

:::{tip}
WGMMA는 흔히 **다단계 파이프라인**과 결합합니다. 텐서 코어가 타일 *k*를 처리하는 동안 TMA는 두 번째 공유 메모리 버퍼에 타일 *k+1*을 불러옵니다. TMA 배리어와 MMA 배리어는 서로 달라 데이터 이동과 연산을 완전히 겹칠 수 있습니다.
:::

---

## tcgen05 — Blackwell (SM 100)

Blackwell은 **tcgen05**라는 근본적으로 다른 행렬 가속기를 도입합니다. 핵심 혁신은 다음과 같습니다.

1. **단일 스레드 발행.** 워프그룹 집합 명령 대신 스레드 하나가 MMA를 발행하고 하드웨어가 내부적으로 작업을 분산합니다.
2. **텐서 메모리(TMEM).** 레지스터 파일과 분리된 누산기 전용 온칩 메모리입니다. TMEM은 레지스터보다 크며 접근 특성도 다릅니다.
3. **더 큰 타일.** tcgen05는 최대 256×256 모양을 지원합니다. WGMMA는 64×256까지 지원합니다.

### TMEM — 텐서 메모리

TMEM은 레지스터 파일과 나란히 있는 새로운 메모리 계층이며 행렬 누산을 전담합니다. 명시적으로 할당하고 해제해야 합니다.```rust
use cuda_device::SharedArray;
use cuda_device::tcgen05::{TmemGuard, TmemReady, TmemUninit};

static mut TMEM_ADDR: SharedArray<u32, 1> = SharedArray::UNINIT;

// Allocate TMEM (warp-collective)
let tmem: TmemGuard<TmemReady, 128> = unsafe {
    TmemGuard::<TmemUninit, 128>::from_static(TMEM_ADDR.as_mut_ptr())
        .alloc()
};

// ... use tmem for MMA ...

// Deallocate (warp-collective, required before kernel exit)
unsafe { tmem.dealloc(); }
````TmemGuard`는 타입 상태를 사용합니다: `TmemUninit` → `alloc()` → `TmemReady` → `dealloc()` → `TmemDeallocated`. 타입 시스템이 할당 전에 TMEM을 사용하는 것과 해제를 잊는 것을 막습니다. 해제하지 않으면 하드웨어 리소스가 누수되고 오류가 발생합니다.

`N_COLS` const 매개변수는 TMEM 타일 폭을 결정합니다. 일반적인 구성은 다음과 같습니다.

| `N_COLS` | 워프당 TMEM | 사용 사례 |
|:---:|:---|:---|
| 64 | 가장 작은 할당 | 좁은 타일 |
| 128 | 기본값 | 표준 GEMM 타일 |
| 256 | 최댓값 | 넓은 타일, 더 높은 처리량 |

### 명령 설명자와 SMEM 설명자

tcgen05는 MMA 명령 하나에 설명자 두 개를 사용합니다.

**명령 설명자** — MMA 구성을 인코딩합니다.```rust
use cuda_device::tcgen05::{Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape};

let idesc = Tcgen05InstructionDescriptor::builder()
    .shape(Tcgen05MmaShape::M128_N128)
    .a_type(Tcgen05ElementType::F16)
    .b_type(Tcgen05ElementType::F16)
    .build();
```**SMEM 설명자** — 공유 메모리의 피연산자 타일을 가리킵니다.```rust
use cuda_device::tcgen05::{Tcgen05SmemDescriptor, Tcgen05SwizzleMode};

let a_desc = Tcgen05SmemDescriptor::for_k_major(
    smem_a_addr,
    m, k,
    2, // bytes per element (f16)
    Tcgen05SwizzleMode::Swizzle128B,
);
```### MMA 발행

스레드 하나가 MMA 명령을 발행한 뒤 모든 스레드가 배리어에서 대기합니다.```rust
use cuda_device::tcgen05::{tcgen05_commit, tcgen05_mma_f16};

if thread::threadIdx_x() == 0 {
    unsafe {
        tcgen05_mma_f16(
            tmem.raw_address(),
            a_desc.raw(),
            b_desc.raw(),
            idesc.raw(),
            true, // enable accumulation into D
        );
        tcgen05_commit(mma_barrier_ptr);
    }
}

// All threads wait for MMA completion
mma_barrier.wait(token);
```### 에필로그 — TMEM 결과 읽기

MMA 루프가 끝나면 누산기는 TMEM에 있습니다. 다음 TMA 전역 저장을 위해 공유 메모리로 옮기려면 TMEM에서 레지스터로 읽은 뒤 `stmatrix`로 공유 메모리에 써야 합니다.```rust
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::tcgen05::{
    TmemF32x4, stmatrix_m8n8_x2, tcgen05_ld_16x256b_pure, tcgen05_load_wait,
};

unsafe {
    // Load a 16×256-bit slice from TMEM into registers
    let regs: TmemF32x4 = tcgen05_ld_16x256b_pure(tmem.raw_address());
    tcgen05_load_wait();

    // Convert four f32 accumulators into two registers of packed bf16 values.
    let packed0 = cvt_bf16x2_f32(regs[0], regs[1]);
    let packed1 = cvt_bf16x2_f32(regs[2], regs[3]);

    // Store two 8×8 matrices from registers (warp-collective).
    stmatrix_m8n8_x2(smem_ptr, packed0, packed1);
}
````stmatrix`는 워프 집합 연산입니다. 스레드 32개 모두 참여하며 각 스레드가 자신의 레지스터 부분을 제공해 공유 메모리 타일을 구성합니다.

### CTA 그룹 2 (CG2)

Blackwell은 **CG2 모드**도 지원합니다. 클러스터의 CTA 두 개가 짝을 이루어 조정된 MMA 명령을 발행합니다. 유효 타일 폭이 두 배가 되며 `_cg2` API 변형을 사용해야 합니다.```rust
use cuda_device::tcgen05::{tcgen05_alloc_cg2, tcgen05_mma_f16_cg2};
```CG2와 표준 CG1을 커널 하나에서 섞어 사용할 수 없습니다.

:::{tip}
tcgen05는 **Blackwell 데이터센터 GPU**(SM 100a)에서만 사용할 수 있습니다. 소비자용 Blackwell(SM 120)은 이전 `mma.sync` 명령어 집합을 사용합니다. tcgen05 API를 사용하기 전에 대상 아키텍처를 확인하세요.
:::

---

## 적합한 가속기 선택

| 기능 | WGMMA (Hopper) | tcgen05 (Blackwell DC) | mma.sync (Ampere/소비자용) |
|:---|:---|:---|:---|
| 발행 모델 | 워프그룹 (스레드 128개) | 단일 스레드 | 워프 (스레드 32개) |
| 누산기 | 레지스터 | TMEM (전용) | 레지스터 |
| 최대 타일 (M×N) | 64×256 | 256×256 | 16×8 |
| 비동기 실행 | 가능 (커밋 + 배리어) | 가능 (커밋 + 배리어) | 동기식 |
| TMA 통합 | 기본 지원 | 기본 지원 + CG2 멀티캐스트 | 수동 로드 |
| 클러스터 지원 | 가능 | 가능 + CG2 | 불가 |
| 최소 SM | 90 (Hopper) | 100a (Blackwell DC) | 80 (Ampere) |

대부분 사용자는 대상 GPU에 따라 선택하면 됩니다. 여러 아키텍처를 대상으로 라이브러리를 작성한다면 컴파일 시점 기능 게이트나 런타임 아키텍처 탐지로 실행 경로를 선택해야 합니다.

---

## GEMM의 단계별 구성

이 가속기들은 각각 따로 작동하지 않습니다. 고성능 GEMM 커널은 이 장과 앞선 장의 기법을 결합합니다.

1. **TMA**가 스레드별 로드 없이 타일을 전역 메모리에서 공유 메모리로 가져옵니다.
2. **배리어**가 단계별 TMA 완료를 추적합니다.
3. **MMA 명령**(WGMMA 또는 tcgen05)이 공유 메모리 타일을 처리합니다.
4. **다단계 파이프라인**이 타일 *k*의 MMA와 타일 *k+1*의 로드를 겹칩니다.
5. **워프 특화**로 일부 워프는 로드, 다른 워프는 MMA를 담당합니다.
6. **에필로그**(tcgen05)는 TMEM에서 읽고 정밀도를 변환한 뒤 TMA로 저장합니다.

cuBLAS와 CUTLASS는 내부에서 이 커널 구조를 사용합니다. cuda-oxide에서는 타일에 `SharedArray`, 동기화에 `ManagedBarrier`, 연산에 MMA API를 사용해 같은 구조를 Rust로 구현할 수 있습니다.

[`gemm_sol` 예제](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol)는 구현을 끝까지 보여주는 참고 자료입니다. 4단계 `cta_group::2` 파이프라인은 B200(148 SM)에서 크기 4096³일 때 **868 TFLOPS, `cublasLtMatmul` SoL의 57.8%**에 도달합니다. 더 작은 Blackwell 데이터센터 SKU에서는 SM 수에 따라 처리량이 줄어듭니다(예: 80-SM 변형에서 같은 커널은 약 204 TFLOPS, SoL 약 46%). 두 구성의 단계별 표는 예제 README를 참조하세요. 예제는 `bench/cublaslt_bench`로 실행 중 cublasLt 기준값을 측정하므로 “SoL 대비 비율”은 고정 B200 수치가 아니라 호스트 GPU의 cublasLt 최고치에 대한 비율입니다.

:::{seealso}
- MMA 피연산자 타일 관리는 [공유 메모리와 동기화](https://nvidia.github.io/cuda-rust/cuda-oxide/advanced/shared-memory-and-synchronization.html)를 참조하세요.
- 텐서 코어로 타일을 전달하는 [텐서 메모리 가속기](tensor-memory-accelerator.md)를 참조하세요.
- CG2 모드와 멀티캐스트를 사용하는 다중 CTA MMA는 [클러스터 프로그래밍](cluster-programming.md)을 참조하세요.
:::

# 아키텍처 개요

Rust로 `#[kernel]` 함수를 작성하고 타입 검사와 빌림 검사를 마치면 GPU에서 실행할 PTX로 바꿔야 합니다. 이 장에서는 cuda-oxide의 컴파일 단계, 크레이트, 설계 이유를 살펴봅니다. 커널을 작성하기만 한다면 선택적으로 읽어도 됩니다.

## 설계 철학

핵심은 **각 단계에 적합한 도구를 선택하되 파이프라인 전체는 직접 관리한다**는 것입니다.

- **프런트엔드: `rustc`와 `rustc_public`(Stable MIR).** 파싱, 이름·타입 해석, 빌림 검사, 트레이트 해석, 단형화, MIR 최적화를 Rust 컴파일러에 맡깁니다.
- **미들엔드: `pliron`(MLIR 유사 IR).** 순수 Rust 프레임워크로 MIR을 LLVM 형태로 변환합니다. C++/CMake/tablegen 의존성 없이 MIR 및 NVIDIA 인트린식 dialect를 정의하고 upstream `pliron-llvm`의 LLVM dialect를 사용합니다.
- **백엔드: LLVM NVPTX.** LLVM의 NVPTX 백엔드가 PTX 명령 인코딩, 레지스터, 스케줄링을 처리합니다. LLVM IR을 텍스트로 내보내 `llc`로 PTX를 생성합니다.

따라서 마지막 `llc` 호출 외에는 컴파일러가 Rust로 작성되어 있습니다. 내부 단계에 중단점을 설정하고 IR을 출력하거나 Miri로 실행할 수 있습니다.

:::{note}
외부 바이너리는 `llc` 하나입니다. Rust 도구 체인의 `llvm-tools` 구성 요소에 NVPTX 백엔드를 포함한 `llc`가 들어 있으며 `rust-toolchain.toml`에 지정되어 있습니다. 시스템 LLVM 21 이상도 사용할 수 있습니다. CUDA Toolkit에는 `llc`가 없습니다. LLVM IR까지는 Rust 코드가 처리하고 이후 `llc`가 PTX를 생성합니다.
:::

## 파이프라인 개요

```{figure} images/compiler-pipeline.svg
:align: center
:width: 100%

Rust 소스가 rustc 프런트엔드와 Stable MIR을 거쳐 `dialect-mir`로 변환됩니다. `mem2reg`가 alloca를 SSA로 승격하고, 주석 기반 루프 언롤링, LLVM dialect, 텍스트 LLVM IR을 거쳐 PTX가 됩니다.
```

1. **Rust 소스:** `#[kernel]` 매크로가 함수를 `cuda_oxide_kernel_<hash>_<name>` 예약 네임스페이스로 바꿉니다. 접두사는 `reserved-oxide-symbols`에 있고 해시가 충돌을 방지합니다.
2. **rustc:** 파싱, 타입·빌림 검사, 단형화와 인라이닝·상수 전파·데드 코드 제거를 수행합니다.
3. **Stable MIR:** 코드 생성 백엔드가 rustc 내부 MIR을 `rustc_public` 타입으로 변환합니다.
4. **`dialect-mir`:** `mir-importer`가 place, projection, `Rvalue`, `BinOp` 등을 pliron dialect로 가져옵니다. 로컬별 `mir.alloca`, 블록 간 `mir.load`/`mir.store`를 `pliron::opts::mem2reg`가 SSA 값으로 승격합니다. 이후 [컴파일러 최적화](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/compiler-optimizations.html)가 실행됩니다.
5. **LLVM dialect:** `mir-lower`가 `llvm.alloca`, `llvm.load`, `llvm.store`, `llvm.getelementptr`, `llvm.call` 등으로 Rust 개념을 기계 지향 IR로 낮춥니다.
6. **LLVM IR:** NVVM 빌드는 `nvvm-transforms`로 libNVVM이 받는 형태로 바꿉니다. 일반 PTX 빌드는 건너뛰며 `llvm-export`가 텍스트 `.ll`을 출력합니다.
7. **GPU 산출물:** 일반 빌드는 `llc`로 PTX를 생성하고 NVVM 빌드는 libNVVM/nvJitLink로 LTOIR 및 cubin을 만듭니다.

## 크레이트 구성

| 크레이트 | 역할 |
|:--|:--|
| `rustc-codegen-cuda` | rustc 백엔드. `codegen_crate()`를 가로채 호스트/디바이스 코드를 분리합니다. |
| `mir-importer`, `dialect-mir` | Stable MIR을 Rust 의미를 모델링한 pliron dialect로 변환합니다. |
| `mir-transforms`, `mir-lower` | MIR 최적화와 LLVM dialect 변환을 수행합니다. |
| `nvvm-transforms`, `llvm-export` | NVVM 적합 변환 및 LLVM IR 내보내기입니다. |
| `dialect-nvvm` | `tid`, `ntid`, 배리어, TMA 등 NVIDIA 인트린식 dialect입니다. |
| `cargo-oxide` | `cargo oxide build`, `run`, `pipeline` CLI입니다. |
| `cuda-device`, `cuda-macros` | 디바이스 API와 `#[kernel]`, `#[device]` 매크로입니다. |
| `cuda-host`, `cuda-core`, `cuda-core-derive` | 모듈 로드, Driver API, `DeviceCopy` derive입니다. |
| `cuda-async`, `cuda-bindings` | `DeviceOperation` 조합기와 CUDA Driver FFI입니다. |
| `cuda-intrinsics`, `cuda-intrinsics-gen` | 인트린식 선언, 추출, 결정적 생성 도구입니다. |
| `cuda-artifact-finalizer`, `oxide-artifacts` | NVVM IR/LTOIR/PTX 마무리 및 임베디드 산출물 메타데이터입니다. |
| `oxide-tile-export`, `oxide-tilebc` | Tile IR 상호 운용 export와 바이트코드 작성기(미출시 작업)입니다. |
| `dialect-iket`, `iket-lower` | 커널 내부 이벤트 추적 및 계측 lowering입니다. |
| `dialect-ptx`, `ptx-parse`, `ptx-schedule` | 구조화 PTX, 구문 분석 뷰, 스케줄 교란 퍼징입니다. |
| `cuda-oxide-codegen` | rustc 독립 PTX 백엔드입니다. |
| `libnvvm-sys`, `nvjitlink-sys` | `dlopen` 기반 libNVVM 및 nvJitLink 바인딩입니다. |
| `cuda-target-spec`, `reserved-oxide-symbols` | CUDA 대상 정책과 내부 심볼 이름 규약입니다. |
| `fuzzer` | rustlantis 기반 차등 코드 생성 퍼저입니다. |

### 의존성 흐름

```{figure} images/crate-dependency-flow.svg
:align: center
:width: 100%

백엔드, importer, lowering 패스가 차례로 연결됩니다. dialect 크레이트는 pliron을 공통 기반으로 사용합니다.
```

pliron은 `Context`, `Module`, `Region`, `Block`, `Operation`, `Type`, `Attribute` 기반을 제공합니다. LLVM dialect는 upstream `pliron-llvm`에서 오며 `llvm-export`가 텍스트 `.ll` 출력기를 더합니다. `rustc_public`은 importer가 읽는 안정 MIR 타입을 제공합니다. 사용자용 `cuda-device`, `cuda-macros`, `cuda-host`, `cuda-core`, `cuda-async`는 컴파일러 내부와 독립적입니다.

## 핵심 외부 의존성

### pliron — MLIR 유사 Pliron IR

[pliron](https://github.com/vaivaswatha/pliron)은 LLVM MLIR에서 영감을 받은 순수 Rust 확장형 컴파일러 IR입니다. dialect, 연산, 타입, 속성, 영역, 블록을 제공하며 C++ 도구 체인, CMake, tablegen을 요구하지 않습니다. cuda-oxide는 컴파일러 전체를 `cargo`로 빌드하기 위해 MLIR 대신 pliron을 선택했습니다. `dialect-mir`와 `dialect-nvvm`은 lowering 경로에 있고 `dialect-iket`, `dialect-ptx`는 각각 커널 이벤트 추적과 구조화 PTX에 사용됩니다.

:::{seealso}
자세한 내용은 [Pliron — Rust의 MLIR](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/pliron.html)을 참조하세요.
:::

### rustc_public — Stable MIR

`rustc_public`은 Rust 컴파일러 내부에 접근하는 안정 API입니다. MIR은 빌림 검사와 최적화가 수행되고 타입 정보를 보존해 GPU 백엔드의 좋은 출발점이 됩니다. MIR 자체는 내부 표현으로 나이틀리마다 바뀔 수 있으므로 이 API가 내부 타입을 공개 인터페이스로 연결합니다. cuda-oxide는 `CodegenBackend::codegen_crate()`에서 안정 MIR로 변환해 `mir-importer`에 전달합니다.

:::{seealso}
자세한 내용은 [rustc_public — Stable MIR](rustc-public.md)을 참조하세요.
:::

## 호스트와 디바이스 분리

cuda-oxide는 호스트와 디바이스 코드가 같은 `.rs` 파일에 있는 단일 소스 컴파일러입니다.

1. `cargo-oxide`가 사용자 지정 백엔드로 rustc를 실행합니다.

   ```bash
   cargo oxide run vecadd
   ```

   내부적으로 `-Z codegen-backend=librustc_codegen_cuda.so`를 설정합니다.
2. rustc가 의존성의 모든 크레이트에 `codegen_crate()`를 호출합니다.
3. 백엔드가 `#[kernel]` 매크로가 만든 `cuda_oxide_kernel_<hash>_` 접두사의 함수를 찾습니다.
4. 커널이 있으면 전이 호출 그래프에서 디바이스 함수를 수집하고 `mir-importer`가 `dialect-mir` → LLVM dialect → `.ll` → PTX로 변환합니다.
5. 호스트 코드는 언제나 표준 LLVM 백엔드에서 컴파일됩니다.
6. 빌드 결과는 호스트 바이너리와 PTX입니다.

```text
target/debug/vecadd          ← 실행 중 PTX를 로드하는 호스트 바이너리
target/debug/vecadd.ptx      ← CUDA 드라이버가 로드하는 디바이스 코드
```

:::{note}
`cuda-device` 같은 의존성의 디바이스 코드는 지연 컴파일됩니다. 사용자 크레이트의 커널에서 전이 호출될 때만 PTX로 만들며 `.rlib` 메타데이터의 Stable MIR을 사용하므로 소스에서 의존성을 다시 빌드할 필요가 없습니다.
:::

### 간단한 모델

```{figure} images/host-device-split.svg
:align: center
:width: 100%

빌드 한 번으로 두 대상을 컴파일합니다. 모든 함수가 rustc 프런트엔드를 지나고, 커널은 cuda-oxide로 나머지는 LLVM으로 전달됩니다.
```

백엔드는 함수가 커널이거나 커널에서 호출되는지 확인합니다. 해당하면 cuda-oxide 경로, 아니면 LLVM 경로를 택합니다. 양쪽에서 쓰이는 제네릭 도우미는 호스트와 디바이스에서 각각 컴파일될 수 있습니다.

## rustc가 제공하는 기능

| 기능 | GPU 코드에서의 이점 |
|:--|:--|
| 타입 검사 | GPU 컴파일 전 오류를 찾아 난해한 PTX 오류를 피합니다. |
| 수명 추적 | 호스트/디바이스 경계를 포함해 수명을 확인합니다. |
| 빌림 검사 | Rust 별칭 규칙에 GPU 전역 쓰기용 `DisjointSlice`와 실행 증명을 더합니다. |
| 단형화 | 제네릭을 구체화해 `map<f32, _>`도 실제 커널로 만듭니다. |
| MIR 최적화 | 인라이닝, 상수 전파, 데드 코드 제거를 미리 적용합니다. |
| 트레이트 해석 | 트레이트 객체와 vtable을 제거해 정적 디스패치로 만듭니다. |
| 패턴 매칭 | `match`를 최적화된 `SwitchInt` MIR 종결자로 바꿉니다. |

cuda-oxide는 최적화·단형화·빌림 검사가 끝난 MIR을 번역합니다. 커널 타입 오류 역시 제안과 위치 강조가 있는 일반 rustc 진단으로 표시됩니다.

## 다음에 읽을 내용

- [Pliron](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/pliron.html): IR 프레임워크
- [rustc_public — Stable MIR](rustc-public.md): 안정 MIR 인터페이스
- [rustc-codegen-cuda](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/rustc-codegen-cuda.html): 코드 생성 백엔드
- [MIR importer](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/mir-importer.html): Stable MIR 가져오기
- [컴파일러 최적화](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/compiler-optimizations.html): MIR 분석 및 변환
- [Pliron dialect](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/mlir-dialects.html): 커스텀 dialect
- [낮춤 파이프라인](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/lowering-pipeline.html): MIR에서 LLVM dialect로
- [새 인트린식 추가](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/adding-new-intrinsics.html): 확장 기여 안내

# 소스에서 빌드하기

이 부록에서는 새로 체크아웃한 저장소에서 cuda-oxide 개발 환경을 준비하는 방법을 설명합니다. 예제만 실행하려면 [첫 GPU 커널 작성하기](../getting-started/hello-gpu.md) 장을 따라가는 편이 더 빠릅니다.

---

## 요구사항

| 의존성 | 버전 | 용도 |
|:---|:---|:---|
| **Rust nightly** | `nightly-2026-10-11` (고정) | 코드 생성 백엔드에 필요한 `rustc-dev`가 포함된 컴파일러 도구 체인 |
| **CUDA Toolkit** | 13.0 이상 (cuRAND 헤더 포함) | Driver API, `nvcc`, PTX 어셈블러. bindgen에 `curand.h` 필요 |
| **Clang** | 21 이상 (`clang-21` 패키지) | 호스트 `cuda-bindings`의 `bindgen`이 Clang 헤더를 사용 |
| **Linux** | Ubuntu 24.04에서 검증 | Windows와 macOS는 지원하지 않음 |
| **GPU** | sm_80, sm_90, sm_100a | 하드웨어 대상 |

## 저장소 복제

```bash
git clone https://github.com/NVIDIA/cuda-rust.git
cd cuda-rust
```

## Rust 도구 체인 설치

CUDA Oxide 구성요소에는 정확한 Nightly 버전과 구성요소를 고정한 `cuda-oxide/rust-toolchain.toml`이 있습니다. `cuda-oxide/` 안에서 실행하는 명령에는 rustup이 이 설정을 자동 적용합니다. 저장소 루트와 `cutile-rs/`는 stable을 사용합니다.

```toml
# cuda-oxide/rust-toolchain.toml
[toolchain]
channel = "nightly-2026-10-11"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
```

직접 설치해야 한다면 다음을 실행합니다.

```bash
rustup toolchain install nightly-2026-10-11
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain nightly-2026-10-11
```

`rust-src`는 교차 컴파일을 위한 표준 라이브러리 소스를 제공하고, `rustc-dev`는 코드 생성 백엔드가 링크하는 컴파일러 내부 구성요소를 노출합니다. `llvm-tools`는 PTX 생성에 사용하는 도구 체인 포함 `llc`를 설치하며 `cargo oxide doctor`에도 필요합니다. 나머지 세 구성요소는 빌드에 필요하지 않습니다. `rust-analyzer`는 IDE 지원, `clippy`는 CI 린트 검사, `rustfmt`는 `cargo oxide fmt`를 제공합니다.

## CUDA 설치

CUDA Toolkit이 `PATH`에 있는지 확인합니다.

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version   # 13.x 이상이 표시되어야 합니다.
```

GPU가 없는 시스템(예: CI)에서 빌드하더라도 `ptxas`와 헤더 파일 때문에 Toolkit이 필요합니다. 다만 커널을 실행할 수는 없습니다.

## LLVM 설치 (일반적으로 선택 사항)

코드 생성 파이프라인은 LLVM IR을 출력하고 `llc`를 호출해 PTX를 만듭니다. 고정된 Rust 도구 체인(`nightly-2026-10-11`)은 `llvm-tools` 구성요소를 통해 NVPTX 백엔드가 활성화된 LLVM 23을 이미 제공합니다. 권장 경로는 다음과 같습니다.

```bash
rustup component add llvm-tools
```

이 구성요소는 이미 `rust-toolchain.toml`에 포함되어 있으므로 새 체크아웃에서는 rustup이 자동 설치합니다. 오래된 체크아웃에서는 위 명령을 한 번 실행하면 됩니다. 파이프라인은 `<sysroot>/lib/rustlib/<host>/bin/llc`에서 이 바이너리를 자동으로 찾습니다.

특정 패치 버전을 사용하거나 이미 시스템 LLVM을 설치한 경우 파이프라인은 대신 `PATH`의 `llc-23` / `llc-22` / `llc-21`을 탐색합니다. 최소 버전은 LLVM 21입니다. 이전 버전은 cuda-oxide가 생성하는 TMA / tcgen05 / WGMMA intrinsic의 서명을 처리하지 못합니다.

```bash
# Ubuntu / Debian
sudo apt install llvm-21
```

배포판에서 `llvm-21`을 제공하지 않는다면 LLVM apt 도우미를 사용합니다.

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

```bash
# NVPTX 지원 확인
llc-21 --version | grep nvptx
```

대상 목록에 `nvptx64 - NVIDIA PTX 64-bit`가 표시되어야 합니다.

rustup, 배포판 또는 사용자 정의 빌드의 특정 바이너리를 고정하려면 `CUDA_OXIDE_LLC=/path/to/llc`를 설정합니다. 파이프라인은 다음 순서로 바이너리를 찾습니다.

1. `$CUDA_OXIDE_LLC` (설정된 경우)
2. Rust 도구 체인의 `llvm-tools`가 제공하는 `llc`
3. `PATH`에서 `llc-23`, `llc-22`, `llc-21`, 버전 없는 `llc` 순으로 탐색

```{note}
LLVM 20 이하의 오래된 `llc`도 `CUDA_OXIDE_LLC=/path/to/llc-20`으로 지정하면 단순한 커널을 컴파일할 수 있습니다. 하지만 최신 TMA / tcgen05 / WGMMA intrinsic(`tma_copy`, `gemm_sol`, `tcgen05_matmul`, `wgmma` 등)을 사용하는 예제는 LLVM 21 이상으로 업그레이드할 때까지 `Intrinsic has incorrect argument type!` 오류로 실패합니다.
```

## Clang 설치 (호스트 `cuda-bindings`용)

호스트 `cuda-bindings` 크레이트는 `bindgen`을 실행합니다. bindgen은 libclang을 불러오고 Clang 자체 resource-dir의 `stddef.h`를 필요로 하므로 `libclang1-*` 런타임만으로는 부족합니다.

```bash
sudo apt install clang-21   # 또는 libclang-common-21-dev
```

`cargo oxide doctor`가 이 항목을 사전에 검사합니다.

## 작업공간 빌드

SIMT 작업공간은 `cuda-oxide/`에 있습니다. 사용자용 크레이트(`cuda-device`, `cuda-host`, `cuda-macros` 등)와 빌드 도구(`cargo-oxide`)를 포함합니다. 호스트 런타임(`cuda-bindings`, `cuda-core`, `cuda-async`)은 cutile-rs와 공유하며 저장소 루트에 있습니다. SIMT 작업공간은 이 크레이트들을 경로로 의존합니다. 이 절, “설치 확인”, “일반 명령”은 Nightly 고정 버전과 `cargo oxide` 작업공간 별칭을 적용하는 `cuda-oxide/`에서 실행합니다.

```bash
cd cuda-oxide
cargo build
```

```{note}
코드 생성 백엔드(`cuda-oxide/crates/rustc-codegen-cuda/`)는 특수 Nightly 기능과 별도 빌드 과정이 필요하므로 의도적으로 작업공간 구성원이 아닙니다. `cargo-oxide`가 이를 투명하게 빌드합니다.
```

## cargo-oxide 설치

`cargo-oxide`는 전체 컴파일 파이프라인을 구동하는 Cargo 하위 명령입니다. 저장소 안에서는 작업공간 별칭으로 동작합니다. 독립적으로 사용하려면 고정된 Nightly 도구 체인으로 설치합니다.

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
```

첫 실행 시 코드 생성 백엔드 dylib를 자동으로 가져와 빌드합니다.

## 설치 확인

`cuda-oxide/`에서 다음을 실행합니다.

```bash
# 모든 사전 요구사항 점검
cargo oxide doctor

# 기본 첫 예제를 컴파일하고 실행
cargo oxide run vecadd
```

`cargo oxide doctor`는 Rust 도구 체인, CUDA Toolkit(수학 intrinsic을 사용하는 커널용 libNVVM / nvJitLink / libdevice 포함), LLVM 설치, 코드 생성 백엔드를 검사합니다. 설정이 올바르면 `cargo oxide run vecadd`가 Rust 커널을 PTX로 컴파일하고 GPU에서 실행해 성공 메시지를 출력합니다.

## 자주 쓰는 명령

`cuda-oxide/`에서 실행합니다.

```bash
# 예제를 빌드하고 실행
cargo oxide run <example>

# 생성된 PTX만 출력
cargo oxide inspect <example>

# 전체 컴파일 파이프라인 표시 (MIR → LLVM IR → PTX)
cargo oxide pipeline <example>

# 로컬 빌드 출력과 생성 산출물 삭제
cargo oxide clean

# NVIDIA Compute Sanitizer로 검사
cargo oxide sanitize <example> --tool memcheck

# cuda-gdb로 디버깅
cargo oxide debug <example> --tui

# libNVVM/nvJitLink 상호 운용을 위한 NVVM IR 빌드
cargo oxide build <example> --emit-nvvm-ir --arch sm_120
```

## 책 빌드

문서는 `cuda-oxide-book/`에 있으며 Sphinx와 MyST Markdown을 사용합니다. 로컬에서 빌드하고 서버를 실행하려면 다음을 따릅니다.

```bash
# 저장소 루트에서 실행:
cd cuda-oxide/cuda-oxide-book
make setup      # 가상 환경 생성 및 의존성 설치
source .venv/bin/activate
make livehtml   # http://localhost:8000 에 개발 서버 실행
```

## API 문서 생성

작업공간 크레이트에는 일반 `cargo doc`을 사용할 수 있습니다.

```bash
# 저장소 루트에서 실행:
cd cuda-oxide
cargo doc --no-deps --open
```

이 명령은 `cuda-device`, `cuda-host`, `cuda-macros`와 다른 작업공간 구성요소의 rustdoc을 생성합니다. 작업공간 구성원이 아닌 코드 생성 백엔드는 제외됩니다. 공유 호스트 런타임(`cuda-core`, `cuda-async`)은 저장소 루트 작업공간에 있으므로 문서를 만들려면 해당 루트에서 `cargo doc`을 실행하세요.

## 작업공간 구조

```text
repository/
├── Cargo.toml              # Stable 공유 호스트 작업공간
├── cuda-bindings/          # 공유 CUDA 바인딩
├── cuda-core/              # 공유 호스트 런타임
├── cuda-core-derive/       # cuda-core 파생 매크로
├── cuda-async/             # 공유 비동기 런타임
├── cutile-rs/              # 타일 DSL (중첩 가져오기)
└── cuda-oxide/
    ├── Cargo.toml          # Nightly SIMT 작업공간
    ├── rust-toolchain.toml # 고정 Nightly + 구성요소
    ├── crates/
    │   ├── cuda-device/          # 장치 intrinsic (#![no_std])
    │   ├── cuda-host/            # 호스트 실행 API
    │   ├── cuda-macros/          # 프로시저 매크로 (#[kernel], #[device], gpu_printf!)
    │   │                         # (공유 호스트 크레이트는 git 루트에 위치;
    │   │                         #  SIMT API는 해당 크레이트의 simt 모듈에 위치)
    │   ├── cargo-oxide/          # Cargo 하위 명령
    │   ├── rustc-codegen-cuda/   # 코드 생성 백엔드 (작업공간 구성원 아님)
    │   ├── mir-importer/         # MIR → Pliron IR 변환
    │   ├── mir-lower/            # `dialect-mir` → LLVM dialect 낮추기
    │   ├── dialect-mir/          # Rust MIR를 모델링하는 Pliron dialect
    │   ├── llvm-export/          # pliron-llvm 재노출 및 텍스트 .ll 내보내기 shim
    │   ├── dialect-nvvm/         # NVVM intrinsic dialect
    │   ├── libnvvm-sys/          # libNVVM dlopen 바인딩
    │   ├── nvjitlink-sys/        # nvJitLink dlopen 바인딩
    │   ├── reserved-oxide-symbols/ # 공유 명명 규칙
    │   └── fuzzer/               # 차등 테스트 지원
    └── cuda-oxide-book/        # 이 책 (Sphinx + MyST)
```

## 문제 해결

`llc`를 찾을 수 없거나 NVPTX가 없음
: 가장 빠른 해결 방법은 `rustup component add llvm-tools`입니다. 고정 도구 체인의 `llc`는 NVPTX가 활성화된 LLVM 23이며 자동 선택됩니다. 또는 시스템 LLVM 21 이상(`sudo apt install llvm-21`)을 설치하세요. 파이프라인은 rustup `llc`를 먼저 확인한 다음 `PATH`의 `llc-23` → `llc-22` → `llc-21` 순서로 검색합니다. 특정 바이너리를 고정하려면 `CUDA_OXIDE_LLC=/path/to/llc`를 설정하세요.

`llc`에서 `Intrinsic has incorrect argument type!` 오류
: 사용 중인 `llc`가 LLVM 21보다 오래되어 최신 TMA / tcgen05 / WGMMA intrinsic 서명을 낮추지 못합니다. `llvm-21`을 설치하고 다시 실행하세요.

`error[E0463]: can't find crate for rustc_middle`
: `rustc-dev` 구성요소가 빠져 있습니다. `rustup component add rustc-dev --toolchain nightly-2026-10-11`을 실행하세요.

CUDA 드라이버 버전 불일치
: Toolkit 버전(컴파일 시점)과 드라이버 버전(실행 시점)이 호환되어야 합니다. `nvidia-smi`로 드라이버 버전을, `nvcc --version`으로 Toolkit 버전을 확인하세요.

`cargo oxide doctor`가 코드 생성 백엔드에서 실패
: 백엔드는 처음 사용할 때 빌드됩니다. 빌드가 실패하면 `rust-src`가 설치되어 있는지, Nightly 버전이 `rust-toolchain.toml`과 일치하는지 확인하세요.

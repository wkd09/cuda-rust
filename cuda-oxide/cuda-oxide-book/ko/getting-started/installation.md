# 설치하기

이 절에서는 `cargo oxide run vecadd`를 실행하는 데 필요한 모든 환경을 준비합니다.

---

## 사전 요구사항

| 요구사항 | 버전 | 참고 |
|---|---|---|
| **Linux** | Ubuntu 24.04에서 검증 | 다른 배포판도 작동할 수 있지만 검증되지 않았습니다. |
| **NVIDIA GPU** | Ampere 이상 (sm_80+) | 드라이버 580 이상 (CUDA 13.x 드라이버, 실행 시 로드) |
| **CUDA Toolkit** | 13.0 이상 | `nvcc`, `cuda.h`, `curand.h`를 사용할 수 있어야 합니다. |
| **LLVM** | 21 이상 | NVPTX 백엔드를 포함해야 합니다. |
| **Clang** | 21 이상 | `clang-21` — `bindgen`이 호스트 `cuda-bindings`를 생성할 때 필요합니다. |
| **Rust** | 고정된 Nightly | `cuda-oxide/rust-toolchain.toml`에 지정되어 있습니다. |

호스트 런타임은 빌드 시점에 `libcuda`를 링크하지 않습니다. 공유 `cuda-bindings` 크레이트가 첫 드라이버 호출 때 라이브러리를 로드하므로, 드라이버 없이도 바이너리는 시작되지만 해당 호출에서 `CUDA_ERROR_NOT_INITIALIZED`로 실패합니다. 오류 메시지에는 로더가 찾았던 라이브러리 파일이 표시됩니다. 드라이버의 CUDA 주 버전이 Toolkit보다 낮으면(예: CUDA 13.x로 빌드했는데 12.x 드라이버인 경우) 같은 방식으로 실패하며 “driver too old” 메시지가 나옵니다.

:::{note}
cuda-oxide는 현재 **Linux만** 대상으로 합니다. Windows는 지원하지 않습니다.
:::

---

## 개발 컨테이너

저장소에는 `cuda-oxide/.devcontainer/`에 cuda-oxide 개발 컨테이너 설정이 들어 있습니다. CUDA Toolkit 13.0, LLVM 21, Clang 21, 고정된 Rust Nightly가 미리 설치되어 있어 재현 가능한 개발 환경을 가장 빠르게 준비할 수 있습니다.

호스트에는 CUDA Toolkit을 설치할 필요가 없습니다. 다음 항목은 필요합니다.

- NVIDIA GPU
- CUDA 13.0과 호환되는 NVIDIA 드라이버 (R580 이상)
- NVIDIA Container Toolkit이 설치된 Docker

개발 컨테이너를 지원하는 편집기에서 저장소의 `cuda-oxide/` 폴더를 열고 “Reopen in Container”를 선택합니다. 편집기가 `.devcontainer/devcontainer.json`을 읽고 이미지를 빌드하며 `--gpus=all`로 GPU 접근을 요청합니다. 컨테이너는 저장소 전체를 마운트해 공용 호스트 크레이트를 사용할 수 있게 하고 `/workspaces/cuda-rust/cuda-oxide`를 작업 디렉터리로 엽니다.

CLI만 사용하는 경우 저장소 루트에서 실행합니다.

```bash
npx -y @devcontainers/cli up --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json
```

그런 다음 컨테이너 안에서 다음 명령을 실행합니다.

```bash
npx -y @devcontainers/cli exec --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide doctor
npx -y @devcontainers/cli exec --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide run vecadd
```

호스트 드라이버가 너무 오래되면 `nvidia-smi`, `cargo oxide doctor`, `cargo oxide run vecadd` 같은 GPU 명령이 컨테이너 안에서도 실패합니다. 컨테이너 내부의 CUDA Toolkit을 다른 버전으로 설치하지 말고 호스트 NVIDIA 드라이버를 업데이트하세요.

개발 컨테이너를 사용한다면 아래의 CUDA, LLVM, Clang, Rust 수동 설치 절은 건너뛰어도 됩니다.

---

## Nix / flake.nix

저장소에는 재현 가능한 개발 셸(CUDA 13, LLVM 22, Clang, 고정된 Rust Nightly)을 제공하는 `cuda-oxide/flake.nix`도 있습니다. flakes가 활성화된 [Nix](https://nixos.org/download/), 호스트 NVIDIA 드라이버, Linux(x86_64 또는 aarch64)가 필요합니다.

저장소 루트에서 cuda-oxide 작업공간으로 이동해 개발 셸에 진입합니다. 셸에는 `cargo-oxide`가 포함되어 있습니다.

```bash
cd cuda-oxide
nix develop
cargo oxide run vecadd
```

저장소를 복제하지 않고 새 프로젝트를 시작할 수 있습니다.

```bash
nix run 'github:NVIDIA/cuda-rust?dir=cuda-oxide#new' -- my-project
cd my-project && nix develop
```

이 명령은 `cargo oxide new`로 프로젝트 뼈대를 만들고 이 저장소의 개발 셸을 상속하는 `flake.nix`를 추가합니다. 셸 후크는 NixOS와 비 NixOS 시스템에서 호스트 NVIDIA 드라이버 라이브러리를 자동으로 찾습니다. 호스트 드라이버가 너무 오래되었다면 Nix 셸 안의 구성을 바꾸지 말고 드라이버를 업데이트하세요.

기존 프로젝트에 같은 개발 환경을 추가하려면 프로젝트 안에서 실행합니다.

```bash
nix flake init -t 'github:NVIDIA/cuda-rust?dir=cuda-oxide'
nix develop
```

Nix flake를 사용하는 경우 아래의 CUDA, LLVM, Clang, Rust 수동 설치 절은 건너뛰어도 됩니다.

---

## CUDA Toolkit

[NVIDIA CUDA 다운로드](https://developer.nvidia.com/cuda-downloads) 페이지에서 CUDA Toolkit을 설치한 다음 `PATH`에 추가합니다.

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

설치 여부를 확인합니다.

```bash
nvcc --version
```

:::{tip}
CUDA를 기본 경로가 아닌 위치에 설치했다면 `CUDA_TOOLKIT_PATH`를 `include/cuda.h`가 있는 루트 디렉터리로 설정하세요. `CUDA_TOOLKIT_PATH`와 `CUDA_HOME`이 모두 설정되지 않은 경우 아래 경로를 순서대로 확인하며, CUDA 13.0 이상인 `cuda.h`가 처음 발견된 경로를 사용합니다. `/usr/local/cuda-13.3`, `/usr/local/cuda-13.2`, `/usr/local/cuda-13`, `/usr/local/cuda`. Toolkit이 여러 개 설치되어 있다면 가장 높은 버전이 아니라 이 순서가 우선합니다. `/usr/local/cuda` 심볼릭 링크는 마지막입니다. `cargo oxide doctor`는 어떤 루트를 선택했는지, 앞선 경로를 왜 제외했는지 알려줍니다.
:::

(installation-toolkit-driver-compatibility)=

### Toolkit과 드라이버 호환성

CUDA Toolkit과 NVIDIA 드라이버는 별도로 설치됩니다. CUDA 마이너 버전 호환성을 통해 새 Toolkit이 이전 드라이버에서 작동하는 경우가 많지만, 이 호환성은 더 새 버전의 PTX까지 보장하지 않습니다.

```text
cubin ───────────────────────────► 드라이버가 완성된 GPU 코드를 적재
PTX   ──► 드라이버가 컴파일 ─────► 완성된 GPU 코드
```

일반적인 cuda-oxide 경로는 LLVM `llc`에서 PTX를 얻으므로 CUDA Toolkit을 새로 설치해도 PTX 버전 자체는 바뀌지 않습니다. NVVM 정방향 실행 경로는 선택한 Toolkit의 nvJitLink에서 PTX를 얻습니다. 해당 PTX가 드라이버가 이해할 수 있는 버전보다 새로우면 모듈 적재가 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`(오류 222)을 반환합니다.

현재 검증에 사용된 도구 체인에서는 다음 결과가 관찰되었습니다.

```text
LLVM llc ───────────────► PTX 8.7 ──► 드라이버 580 ✓
CUDA 13.3 nvJitLink ───► PTX 9.3 ──► 드라이버 580 ✗ 오류 222
```

이는 검증 당시 관찰된 버전이며 두 경로의 영구적인 특성은 아닙니다. 다른 도구 체인을 진단할 때는 생성된 PTX의 `.version` 지시자를 확인하세요.

CUDA 13.x 마이너 버전 호환성은 드라이버 580부터 제공되며 CUDA 13.3의 전체 기능을 지원하는 해당 드라이버 버전은 610.43.02입니다. 따라서 580 드라이버는 호환되는 완성 cubin은 적재할 수 있지만 CUDA 13.3이 생성한 PTX 9.3을 JIT 컴파일할 수는 없습니다. 최신 드라이버 표는 [CUDA 릴리스 노트](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)를 참조하세요.

드라이버를 업그레이드하거나 해당 명령에 호환되는 Toolkit을 지정하세요.

```bash
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 cargo oxide run <example> --arch sm_86
```

지원 드라이버 범위와 PTX 제한은 NVIDIA의 [마이너 버전 호환성 안내](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)를 확인하세요.

---

## LLVM 21 이상 (선택 사항)

cuda-oxide는 LLVM의 NVPTX 백엔드를 사용해 LLVM IR을 PTX로 낮춥니다.

대개 Rust 도구 체인에 포함된 `llc`만으로 충분합니다.

LLVM 21 이상을 설치하고 `llc-21`(또는 `llc-22`)을 `PATH`에서 사용할 수 있게 하세요.

```bash
# Ubuntu / Debian
sudo apt install llvm-21
```

배포판에서 `llvm-21` 패키지를 제공하지 않는다면 LLVM apt 도우미를 사용합니다.

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

NVPTX 대상이 포함됐는지 확인합니다.

```bash
llc-21 --version | grep nvptx
```

등록된 대상 목록에 `nvptx64`가 표시되어야 합니다. 파이프라인은 `llc-23`, `llc-22`, `llc-21` 순으로 자동 검색합니다. 특정 바이너리를 지정하려면 `CUDA_OXIDE_LLC=/usr/bin/llc-21`로 설정하세요.

:::{warning}
NVPTX 백엔드가 없는 기본 LLVM 빌드는 작동하지 않습니다. Ubuntu의 `llvm-21` 패키지에는 기본 포함되어 있지만 LLVM을 소스에서 빌드한다면 CMake에 `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"`를 전달해야 합니다.
:::

:::{important}
**LLVM 21이 필요한 이유:** cuda-oxide는 LLVM 20 이하의 `llc`가 처리할 수 없는 TMA / tcgen05 / WGMMA intrinsic을 생성합니다. 간단한 커널은 이전 `llc`에서도 동작할 수 있지만(`CUDA_OXIDE_LLC=/path/to/llc-20` 설정), Hopper 또는 Blackwell 기능을 사용하려면 21 이상이 필요합니다.
:::

---

## Clang (호스트 `cuda-bindings`)

호스트 `cuda-bindings` 크레이트는 [`bindgen`](https://github.com/rust-lang/rust-bindgen)을 실행합니다. `bindgen`은 실행 시점에 `libclang.so`를 불러오며 Clang 자체의 resource-dir에 있는 `stddef.h`가 필요합니다. 따라서 `libclang1-*` 런타임만으로는 부족합니다. 다음 세 패키지가 두 요구사항을 모두 충족합니다.

```bash
sudo apt install libclang-21-dev libclang-cpp21-dev libclang-common-21-dev
```

`cargo oxide doctor`가 이 문제를 미리 찾아냅니다. 점검하지 않으면 호스트 빌드 중 `'stddef.h' file not found` 오류가 발생합니다.

:::{tip}
**doctor가 여전히 Clang을 찾지 못하는 경우:**
버전이 붙은 바이너리(예: `clang-21`)는 설치됐지만 bindgen이 요구하는 버전 없는 `clang` 명령으로 연결되지 않았을 수 있습니다. 의존성을 자동으로 관리하는 Clang 메타 패키지를 설치해 해결할 수 있습니다.

```bash
sudo apt install clang libclang-dev
```
:::

:::{note}
**새 Ubuntu 24.04 / DGX-OS:** 위에서 설명한 대로 `apt.llvm.org/llvm.sh`를 통해 LLVM 21을 설치하면 `clang-21` / `clang++-21` 바이너리는 생기지만 `cargo oxide doctor`가 찾는 버전 없는 별칭은 생성되지 않습니다. `update-alternatives`로 추가하세요.

```bash
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-21   100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100
```

Asus GX10 / NVIDIA DGX Spark에서 검증했습니다.
:::

---

## Rust 도구 체인

CUDA Oxide 작업공간의 `cuda-oxide/rust-toolchain.toml`은 정확한 Nightly 버전과 필수 구성요소를 고정합니다. `cuda-oxide/` 안에서 처음 `cargo` 명령을 실행하면 rustup이 올바른 도구 체인을 자동으로 설치합니다. 저장소 루트와 `cutile-rs/`의 명령은 stable을 사용합니다.

직접 설치해야 한다면 다음을 실행합니다.

```bash
rustup toolchain install nightly-2026-10-11
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain nightly-2026-10-11
```

이 가운데 세 구성요소가 코드 생성 백엔드와 doctor에 필요합니다.

- `rust-src` — NVPTX 대상으로 교차 컴파일할 때 필요한 Rust 표준 라이브러리 소스입니다.
- `rustc-dev` — 백엔드가 링크하는 컴파일러 내부 구성요소입니다.
- `llvm-tools` — LLVM IR을 PTX로 낮추는 도구 체인 포함 `llc`를 제공하며 doctor의 최소 버전 검사에도 사용됩니다.

나머지 세 구성요소는 빌드에 필요하지 않습니다. `rust-analyzer`는 IDE 지원, `clippy`는 CI 린트 검사, `rustfmt`는 `cargo oxide fmt`를 제공합니다.

---

## cargo-oxide

`cargo-oxide`는 전체 빌드 파이프라인을 구동하는 Cargo 하위 명령입니다(`cargo oxide run`, `build`, `debug`, `pipeline` 등. 이 절 끝의 명령 참조에서 전체 목록을 확인할 수 있습니다).

**cuda-oxide 저장소 안에서는** 작업공간 별칭을 통해 바로 사용할 수 있어 별도의 설치가 필요하지 않습니다.

**저장소 외부의 프로젝트에서 사용하려면** 고정된 Nightly 도구 체인으로 설치합니다.

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
```

첫 실행 시 `cargo-oxide`는 코드 생성 백엔드를 자동으로 가져와 빌드합니다. 이후에는 캐시된 빌드를 재사용합니다.

```{figure} images/backend-install-by-rename.svg
:align: center
:width: 100%

이름 변경으로 캐시된 백엔드를 설치합니다. 커널을 컴파일하는 모든 `rustc`가 공유 `.so` 하나를 매핑합니다. 설치 프로그램은 새 빌드를 인접한 임시 파일로 복사한 뒤 기존 파일 위에 이름을 바꿔 덮어씁니다. 따라서 실행 중인 컴파일러는 기존 inode를 계속 사용하고, 설치 도중 시작한 컴파일러도 완성된 파일을 엽니다.
```

cuda-oxide 체크아웃에 포함된 예제를 보려면 다음을 실행합니다.

```bash
cargo oxide list
cargo oxide list --json
```

각 예제의 목적과 문서화된 GPU, 아키텍처, CUDA Toolkit 또는 외부 SDK 요구사항을 출력합니다.

커널을 개발할 때 유용한 일상 명령은 다음과 같습니다.

```bash
# 전체 MIR/LLVM 파이프라인 덤프 없이 생성된 PTX 출력
cargo oxide inspect vecadd

# 로컬 target/ 디렉터리와 생성된 장치 산출물(PTX, LLVM IR,
# LTOIR, cubin 및 부가 메타데이터) 제거
cargo oxide clean
```

`inspect`는 `cargo oxide pipeline`의 간략한 대안입니다. `clean`은 프로젝트 로컬 출력만 정리하며 공유 백엔드 캐시 `~/.cargo/cuda-oxide/`는 건드리지 않습니다.

### 명령 참조

`cargo oxide --help`에 표시되는 전체 명령은 다음과 같습니다.

| 명령 | 설명 |
|---|---|
| `run` | 예제 또는 프로젝트를 빌드하고 실행합니다. |
| `sanitize` | NVIDIA Compute Sanitizer로 검사하며 예제 또는 프로젝트를 빌드하고 실행합니다. |
| `build` | 예제 또는 프로젝트를 빌드합니다(컴파일만 하고 실행하지 않음). |
| `fuzz-schedule` | 생성된 PTX의 스케줄을 교란해 스케줄에 민감한 실패를 찾습니다. |
| `test` | cuda-oxide 백엔드로 Cargo 테스트를 실행합니다. |
| `emit-ltoir` | 크레이트의 장치 코드를 한 단계에서 바이너리 LTOIR 산출물로 컴파일합니다. |
| `pipeline` | 전체 컴파일 파이프라인(MIR → PTX/NVVM IR)을 자세한 출력으로 보여줍니다. |
| `debug` | 디버그 정보와 함께 빌드하고 `cuda-gdb`를 실행합니다. |
| `list` | cuda-oxide 작업공간에 포함된 예제를 나열합니다. |
| `inspect` | 예제 또는 프로젝트를 빌드하고 생성된 PTX를 출력합니다. |
| `fmt` | 모든 크레이트(SIMT 작업공간, 코드 생성 백엔드, 예제)를 포맷합니다. |
| `new` | 독립형 cuda-oxide 프로젝트의 뼈대를 만듭니다. |
| `clean` | 프로젝트 로컬 빌드 출력과 생성된 cuda-oxide 산출물을 제거합니다. |
| `doctor` | 환경이 올바르게 설정됐는지 확인합니다. |
| `setup` | 코드 생성 백엔드를 빌드하고 캐시합니다. |
| `update` | 캐시된 코드 생성 백엔드를 새로 고칩니다(작업공간 안에서는 `setup`을 실행). |

네 명령은 Cargo 자체로 할 수 없는 작업을 수행하므로 특히 알아둘 만합니다.

```bash
# cuda-oxide 백엔드로 장치 코드를 컴파일해 테스트 모음 실행
# -- 뒤의 인자는 Cargo에 전달되며, 생략하면 일반 cargo test로 실행
cargo oxide test -- --lib

# SIMT 작업공간, 코드 생성 백엔드, cuda-macros 장치 전용 fixture,
# examples/ 아래의 모든 manifest(중첩된 manifest 포함)를 포맷합니다.
# 각각 별도 [workspace]이므로 루트에서 cargo fmt만 실행하면 대부분 빠집니다.
cargo oxide fmt
cargo oxide fmt --check

# 컴파일러 크레이트 또는 도구 체인 고정값을 변경한 뒤 캐시 백엔드 재빌드
# 작업공간 안에서는 setup을 안내하고 --force를 주면 실제 실행
cargo oxide update

# 타일 또는 C++ 커널이 링크할 LTOIR 생성
# LTOIR은 아키텍처별 산출물이므로 --arch를 반드시 지정
cargo oxide emit-ltoir my_kernels --arch sm_90
```

모든 명령에서 `--help`를 사용하면 해당 명령이 받는 플래그를 확인할 수 있습니다.

---

## 설치 확인

내장 진단을 실행합니다.

```bash
cargo oxide doctor
```

`cargo oxide doctor`는 Rust 도구 체인, CUDA 헤더(`cuda.h`), CUDA Toolkit(수학 intrinsic을 사용하는 커널에 필요한 libNVVM / nvJitLink / libdevice 포함), LLVM 설치, NVIDIA 드라이버/GPU, 코드 생성 백엔드를 한 번에 검사합니다.

`cargo-oxide` 자체는 빌드하거나 실행하는 데 CUDA Toolkit 또는 NVIDIA 드라이버가 필요하지 않습니다. 따라서 아무것도 설치되지 않은 컴퓨터에서도 `doctor`를 실행해 빠진 항목(예: `cuda.h` 또는 드라이버)을 구체적으로 진단할 수 있습니다. 드라이버/GPU 검사는 정보 제공용입니다. GPU가 필요한 것은 `cargo oxide run`뿐이며 `build`와 `pipeline`은 GPU 없이도 동작합니다.

그런 다음 예제를 처음부터 끝까지 빌드하고 실행합니다.

```bash
cargo oxide run vecadd
```

모든 설정이 올바르면 Rust 커널을 PTX로 컴파일해 GPU에서 실행하고 성공 메시지를 출력합니다.

:::{tip}
**자주 발생하는 문제:**

- `No working llc found` — `rustup component add llvm-tools --toolchain nightly-2026-10-11`을 우선 시도하거나 LLVM 21 이상(`sudo apt install llvm-21`)을 설치하고 `/usr/lib/llvm-21/bin`을 `PATH`에 추가하거나 `CUDA_OXIDE_LLC=/usr/bin/llc-21`을 설정하세요.
- 호스트 `cuda-bindings` 빌드 중 `'stddef.h' file not found` — Clang 개발 헤더를 설치하세요: `sudo apt install clang-21`(또는 `libclang-common-21-dev`).
- `cuda.h not found` — `CUDA_TOOLKIT_PATH`를 CUDA 설치 루트로 설정하거나 `/usr/local/cuda/include/cuda.h`가 있는지 확인하세요.
- `rust-src` / `llvm-tools` 구성요소 누락 — `rustup component add rust-src llvm-tools --toolchain nightly-2026-10-11`을 실행하세요.
:::

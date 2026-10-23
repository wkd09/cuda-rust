# Rust와 GPU 생태계

Rust에서 GPU 컴퓨팅을 쉽게 사용할 수 있도록 여러 프로젝트가 활발히 개발되고 있습니다. 그래픽 셰이더, 암시적 오프로딩, CUDA 프로그래밍, 안전한 드라이버 바인딩처럼 각 프로젝트가 서로 다른 문제를 다루며, 생태계 전체도 빠르게 발전하고 있습니다. 이 부록에서는 주변 프로젝트와 비교해 cuda-oxide가 어디에 위치하는지, 서로 어떤 관계인지 간략히 살펴봅니다.

## cuda-oxide의 위치

| 프로젝트 | 접근 방식 | 대상 | 범위 |
|:---|:---|:---|:---|
| **cuda-oxide** | `rustc` 코드 생성 백엔드 | NVIDIA PTX/SASS | 안전한 Rust에서 CUDA 프로그래밍 모델 사용 |
| **Rust-GPU** | `rustc` → SPIR-V | Vulkan/Metal/DX | SPIR-V 기반 그래픽 셰이더 및 컴퓨트 |
| **rust-cuda** | `rustc` → NVVM IR | NVIDIA PTX | NVIDIA GPU에서 Rust 언어 모델 사용 |
| **CubeCL** | 내장 DSL 및 JIT 런타임 | CUDA/ROCm/WGPU | Rust DSL로 여러 GPU 벤더용 컴퓨트 커널 작성 |
| **std::offload** | `rustc` 및 LLVM 오프로딩 | NVIDIA/AMD/Intel | CPU 코드를 암시적으로 오프로딩 |
| **cudarc** | 안전한 CUDA 드라이버 바인딩 | NVIDIA | CUDA 드라이버의 호스트 측 바인딩 |
| **wgpu** | WebGPU API 및 WGSL/Naga | 크로스 플랫폼 | 셰이더 언어를 통한 이식성 있는 컴퓨트 |

표의 “범위”는 각 프로젝트가 중점적으로 최적화하는 설계 방향을 나타내며, 기능의 한계를 뜻하지 않습니다. 일부 프로젝트는 주변 영역에서 겹치므로 하나의 애플리케이션에서 둘 이상을 함께 사용할 수도 있습니다.

## cuda-oxide와 rust-cuda

가장 가까운 이웃이자 cuda-oxide와 자주 혼동되는 프로젝트는 `rust-cuda`입니다. 두 프로젝트 모두 `rustc` 코드 생성 백엔드로 NVIDIA GPU를 대상으로 하므로 얼핏 보면 서로 대체할 수 있어 보입니다. 하지만 설계의 중심은 서로 다른 방향을 향합니다.

- **rust-cuda**는 **NVIDIA GPU에서 Rust를 실행하는 것**에 초점을 둡니다. `async`/`.await` 같은 Rust 편의 기능, 디바이스에서 실행되는 표준 라이브러리 일부, CUDA 개념을 추상화하는 Rust 중심 프로그래밍 모델을 제공합니다.
- **cuda-oxide**는 **Rust에서 CUDA를 사용하는 것**에 초점을 둡니다. 커널 작성, 디바이스 인트린식, SIMT 실행 모델, CUDA 프로그래밍 모델을 안전한 Rust로 직접 표현합니다. GPU에서 실행되는 일반적인 Rust 함수보다는 C++의 `__global__` 함수 작성에 가까운 방향입니다.

두 방향 모두 가치가 있으며 서로 보완적이라고 생각합니다. 두 프로젝트가 발전하는 동안 rust-cuda 유지관리자들과 협력해 왔고 앞으로도 협력을 이어갈 예정입니다.

## 그 밖의 주변 프로젝트

- **Rust-GPU**는 SPIR-V를 통해 그래픽 중심 GPU 프로그래밍을 지원하며 Vulkan, Metal, DirectX에서 사용할 수 있습니다. 컴퓨트 셰이더도 지원하지만 그래픽이 중심입니다. 여러 GPU 벤더에서 실행하거나 셰이더 상호 운용이 필요하다면 Rust-GPU가 적합합니다.
- **CubeCL**은 내장 DSL입니다. Rust 함수에 `#[cube]`를 붙이면 CubeCL 런타임이 요청 시 CUDA, ROCm/HIP, WGSL용 코드로 JIT 컴파일합니다. `rustc` 백엔드는 아닙니다. `#[cube]` 프로시저 매크로가 함수 본문을 CubeCL IR로 변환하고, 런타임이 호스트 GPU에서 사용할 백엔드에 맞춰 낮춥니다. 설계의 중심은 **벤더 간 이식성**입니다. 하나의 커널을 NVIDIA, AMD, WGPU에서 실행할 수 있지만 커널 안에서 임의의 Rust 구문을 사용할 수는 없고 의도적으로 제한된 DSL 범위를 사용합니다. CubeCL은 [Burn](https://github.com/tracel-ai/burn) 머신러닝 프레임워크의 컴퓨트 백엔드이며, Burn이 대표적인 활용 사례입니다. 제한된 DSL로 여러 GPU 벤더에서 같은 커널을 실행해야 한다면 CubeCL을, NVIDIA 하드웨어에서 전체 CUDA 프로그래밍 모델을 안전한 관용적 Rust로 사용하려면 cuda-oxide를 선택할 수 있습니다.
- **std::offload**는 현재 nightly에서 사용할 수 있는 Rust 언어 기능으로, LLVM 오프로딩 런타임을 사용해 CPU 형태의 루프를 가속기로 암시적으로 이동합니다. 사용자는 CPU 스타일 코드를 작성하고 컴파일러와 런타임이 오프로딩을 처리합니다. cuda-oxide는 명시적 실행 모델을 사용하고 offload는 암시적 모델을 사용합니다.
- **cudarc**는 CUDA 드라이버 API의 안전한 Rust 바인딩입니다. 보통 PTX나 CUDA C++로 별도 작성한 커널을 실행하는 호스트 측 라이브러리입니다. cuda-oxide의 호스트 계층(`cuda-core`, cutile-rs와 공유)은 `cuda-host` 실행 매크로와 긴밀하게 통합되어 있지만 cuda-oxide가 생성하는 PTX는 이식성이 있습니다. 따라서 전체 cuda-oxide 스택에 의존하지 않는 프로젝트에서 cuda-oxide 커널을 실행하려면 cudarc도 좋은 선택입니다.
- **wgpu**는 WebGPU API의 Rust 구현입니다. Vulkan, Metal, DirectX 12, 브라우저의 WebGPU를 추상화하는 런타임입니다. 컴퓨트 및 그래픽 셰이더는 WGSL(WebGPU 셰이딩 언어)로 작성하며 Naga가 백엔드별 코드(SPIR-V, MSL, HLSL)로 변환합니다. 두 도구는 스택의 서로 다른 계층에 있습니다. wgpu는 런타임과 셰이더 DSL이고 cuda-oxide는 Rust 자체를 컴파일하는 `rustc` 코드 생성 백엔드입니다.

## 프로젝트에 참여하기

Rust와 GPU를 함께 사용하는 프로젝트를 만들거나 적합한 도구를 평가하고 있다면 의견을 나눠 주세요. 질문, 설계 논의, 소식은 [Discord](https://discord.gg/ZUEr4AhH5C) 커뮤니티에서 확인할 수 있습니다. 팀에는 [GitHub Discussions](https://github.com/NVIDIA/cuda-rust)에서 연락하거나 저장소에 이슈를 열어 문의할 수 있습니다.

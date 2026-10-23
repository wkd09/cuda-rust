# 코드 생성기: rustc-codegen-cuda

Rust 프로그램은 최적화된 MIR을 기계 코드로 바꾸는 코드 생성 백엔드에 도달합니다. 보통 LLVM이 이 역할을 하지만 cuda-oxide는 `rustc-codegen-cuda`를 백엔드로 끼워 넣습니다. 디바이스 코드를 추출해 cuda-oxide 파이프라인으로 보내고 나머지는 아무 일도 없었던 것처럼 LLVM에 위임합니다. 이 장에서는 백엔드 로드, rustc 호출 처리, GPU 대상 함수 수집을 설명합니다.

## rustc가 사용자 백엔드를 로드하는 방법

```bash
rustc -Z codegen-backend=path/to/libfoo.so
```

이 플래그를 전달하면 rustc는 공유 라이브러리를 `dlopen`하고 `dlsym("__rustc_codegen_backend")`으로 진입점을 찾은 다음 `Box<dyn CodegenBackend>` 반환을 기대합니다. 별도 플러그인 등록이나 handshake는 없습니다.

cuda-oxide의 진입점은 다음 형태입니다.

```rust
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    let config = CudaCodegenConfig::from_env();
    let llvm_backend = rustc_codegen_llvm::LlvmCodegenBackend::new();
    Box::new(CudaCodegenBackend { config, llvm_backend })
}
```

`CudaCodegenConfig::from_env()`가 환경 변수(예: `CUDA_OXIDE_VERBOSE`, `CUDA_OXIDE_DUMP_MIR`)를 읽고 표준 LLVM 백엔드를 생성해 `CudaCodegenBackend` 안에 보관합니다. LLVM을 대체하지 않고 감싸는 구조입니다.

:::{note}
`cargo oxide build`가 `-Z codegen-backend` 옵션을 대신 설정하므로 보통 `dlopen` 설정을 직접 쓸 필요가 없습니다.
:::

## 가로채기: `codegen_crate()`

`CodegenBackend`에서 중요한 메서드는 `codegen_crate(tcx)`입니다. rustc가 타입 검사, 빌림 검사, 단형화를 마친 크레이트 전체를 전달해 기계 코드로 만들도록 합니다.

### 1단계: 단형화 항목 수집

```rust
let (items, cgus) = tcx.collect_and_partition_mono_items(());
```

크레이트의 단형화 함수를 코드 생성 단위(CGU)별로 얻습니다. CGU는 병렬 코드 생성에서 객체 파일 하나가 될 함수 묶음입니다.

### 2단계: 디바이스 진입점 찾기

각 CGU의 함수를 순회해 다음 접두사를 찾습니다.

- `cuda_oxide_kernel_<hash>_` — `#[kernel]`이 지정합니다.
- `cuda_oxide_device_<hash>_` — `#[device]`가 지정합니다.

프로시저 매크로와 백엔드 간 계약은 심볼 이름에 담깁니다. 접두사 상수와 일치/제거 함수는 `reserved-oxide-symbols`에 있어 매크로와 수집기가 함께 사용합니다. 8자리 16진수 해시가 우연한 충돌을 사실상 막습니다.

### 3단계: 디바이스 코드가 있으면 수집·컴파일

커널이나 디바이스 함수가 있으면 순서대로 두 작업을 수행합니다.

1. **디바이스 호출 그래프 수집:** `collector::collect_device_functions()`가 각 커널에서 너비 우선 탐색해 디바이스 코드가 전이 호출하는 함수를 찾습니다.
2. **디바이스 코드 생성:** `device_codegen::generate_device_code()`가 수집된 함수를 Stable MIR로 연결하고 `dialect-mir` → `mem2reg` → 표시된 루프 언롤링 → LLVM dialect → `.ll` → PTX 파이프라인을 실행합니다.

### 4단계: 호스트 코드는 항상 컴파일

```rust
self.llvm_backend.codegen_crate(tcx)
```

커널 발견 여부와 무관하게 감싼 LLVM 백엔드가 `main()`, CLI 파서, 비동기 런타임을 포함한 호스트 코드를 모두 컴파일합니다.

### 다른 트레이트 메서드

`join_codegen()`은 병렬 코드 생성 스레드를 기다리고 `link()`는 링커를 실행합니다. 둘 다 변경 없이 LLVM 백엔드에 위임합니다.

```rust
fn join_codegen(&self, ongoing: Box<dyn Any>, sess: &Session) -> ... {
    self.llvm_backend.join_codegen(ongoing, sess)
}

fn link(&self, sess: &Session, codegen: ..., outputs: &OutputFilenames) -> ... {
    self.llvm_backend.link(sess, codegen, outputs)
}
```

`core`, `cuda-core` 등 일반 의존성 크레이트에는 커널이 없으므로 LLVM에 위임만 합니다. 보통 수십 개 크레이트 중 실제 `#[kernel]`을 가진 하나만 디바이스 파이프라인을 활성화합니다.

(rustc-codegen-device-function-collection)=

## 디바이스 함수 수집

커널 진입점을 찾는 것보다 커널이 호출하는 모든 함수를 찾는 일이 더 어렵습니다. 커널은 도우미를, 도우미는 `cuda-device` 인트린식을, 그 인트린식은 `core` 수학 함수를 호출할 수 있습니다. 필요한 함수가 모두 PTX에 있어야 합니다. `collector` 모듈은 MIR 호출 그래프를 너비 우선 탐색합니다.

```text
worklist = [all kernel entry points]
visited  = {}
collected = []

while worklist is not empty:
    fn = worklist.pop()
    if fn in visited: continue
    visited.add(fn)

    mir = tcx.instance_mir(fn)
    collected.push(fn)

    for basic_block in mir:
        for terminator in [Call, Drop]:
            callee = resolve_callee(terminator)
            if should_collect(callee):
                worklist.push(callee)
```

각 함수에서 `tcx.instance_mir()`로 MIR을 얻고 모든 기본 블록의 `Call`, `Drop` 종결자를 조사해 필터 규칙에 맞는 피호출자를 작업 목록에 넣습니다. 결과 `Vec<CollectedFunction>`에는 완전히 구체화된 `Instance`, PTX 메타데이터용 `is_kernel`, PTX에서 보일 `export_name`이 포함됩니다.

### export 이름과 FQDN 정렬

수집기의 export 이름은 MIR 변환기가 호출 대상으로 생성하는 이름과 일치해야 합니다. 양쪽이 크레이트 이름을 포함한 FQDN을 사용합니다. `rustc_public`의 `CrateDef::name()`은 크레이트 이름을 포함하지만 로컬 항목의 `def_path_str()`은 생략하므로 `fqdn()` 도우미가 이를 앞에 붙입니다. lowering 시 정의와 호출 양쪽에서 `::`를 `__`로 바꿔 LLVM/PTX 식별자로 만듭니다. 제네릭이나 클로저처럼 복잡한 이름에는 고유한 mangled 심볼을 사용합니다.

커널 진입점은 별도 경로를 사용합니다. `compute_kernel_export_name`은 `#[kernel]` 기본 이름을 사용하고 제네릭/클로저 제네릭 인스턴스에는 구체적인 함수 항목의 rustc 128비트 타입 ID 해시를 `_TID_<hex32>`로 덧붙입니다. 호스트 실행기도 `core::intrinsics::reflection::type_id`(래퍼 `cuda_host::type_id_u128_of_val`)로 같은 해시를 계산해 통합 빌드 안에서 바이트 단위로 일치시킵니다. 비제네릭 커널은 기본 이름을 그대로 씁니다.

:::{note}
향후 프레임워크를 업그레이드하면 이 FQDN 방식은 충돌을 검사하며 이름을 정리하는 pliron `Legaliser`로 대체될 예정입니다.
:::

### 크레이트 필터 규칙

모든 의존성 함수를 GPU에서 실행할 수 있는 것은 아닙니다.

| 크레이트 | 허용 여부 | 비고 |
|:--|:--|:--|
| 로컬 크레이트 | 허용 | 사용자 커널 |
| `cuda_device` | 허용 | GPU 인트린식; 스텁은 필터링 |
| `core` | 허용 | `no_std` 표준 라이브러리 |
| 다른 `no_std` 크레이트 | 허용 | 의존 트리에 있는 크레이트 |
| `std` | **금지** | 컴파일 오류 |
| `alloc` | GPU 할당기 사용 시 허용 | GPU 할당기가 필요 |

OS 의존성, 파일 I/O, 네트워크가 없는 크레이트는 허용될 수 있지만 `std`를 끌어오면 허용하지 않습니다. 실수로 디바이스 코드에서 `println!`을 사용하면 몇 단계 뒤 난해한 PTX 오류 대신 호출 그래프 분석 시점에 `std::io::_print`가 금지 크레이트에서 왔다는 오류를 냅니다.

### 인트린식 스텁 필터링

`cuda_device`의 `threadIdx_x()` 같은 일부 함수 본문은 `unreachable!()`뿐인 스텁입니다. rustc 타입 검사용 자리 표시자이며 실제 함수로 컴파일되지 않습니다. 수집기는 `cuda_device` 소속이고 유의미한 MIR이 없는 스텁을 건너뜁니다. 이후 importer가 호출을 `nvvm.read_ptx_sreg_tid_x` 같은 하드웨어 연산으로 바꾸므로 스텁은 실행되지 않습니다.

## 크레이트 간 디바이스 컴파일

커널이 `cuda_device::thread::index_1d()`를 호출할 때 사용자 크레이트에 소스가 없어도 MIR을 가져올 수 있는 이유는 rustc가 메타데이터에 MIR을 저장하기 때문입니다.

### `.rlib` 흐름

`cuda_device`에는 보통 `#[kernel]`이 없으므로 백엔드는 LLVM에 위임합니다. 결과 `.rlib`에는 호스트 객체 코드와 타입 정보, 트레이트 구현, 특정 함수의 직렬화된 최적화 MIR을 담은 `.rmeta`가 들어갑니다.

나중에 사용자 `vecadd` 크레이트를 컴파일할 때 collector가 커널을 찾아 BFS로 `cuda_device::thread::index_1d()`에 도달하면 `tcx.instance_mir()`가 `.rmeta`에서 최적화·단형화된 MIR을 가져옵니다.

```{figure} images/cross-crate-rlib.svg
:align: center
:width: 100%

크레이트 간 디바이스 컴파일. 먼저 cuda_device를 직렬화 MIR이 든 `.rmeta`와 함께 `.rlib`로 컴파일합니다. 이후 vecadd의 수집기가 커널에서 cuda_device까지 BFS로 탐색하고 MIR을 읽어 PTX로 컴파일합니다.
```

의존성 함수는 커널이 전이 참조할 때만 **지연 컴파일**됩니다. 호출되지 않는 함수는 의존성에 있어도 PTX에 들어가지 않습니다.

### `.rmeta`에 MIR이 저장되는 함수

rustc는 모든 함수의 MIR을 보존하지 않습니다.

| 범주 | 직렬화 이유 |
|:--|:--|
| 제네릭 함수 | 하위 크레이트에서 단형화할 MIR 템플릿이 필요합니다. |
| `#[inline]`, `#[inline(always)]` | 크레이트 간 인라이닝을 요청했습니다. |
| 작은 leaf 함수 | 호출이 없고 문장이 적어 인라인 비용이 낮다는 휴리스틱입니다. |

`cuda_device` 함수 대부분은 이 범주에 들어갑니다. 이는 cuda-oxide 전용 기능이 아니라 rustc의 크레이트 간 인라이닝 및 제네릭 인스턴스화 동작이며 같은 메커니즘이 `Vec::push` 인라이닝과 디바이스 MIR 획득을 모두 가능하게 합니다.

## Stable MIR 브리지

수집 결과 `Vec<CollectedFunction>`에는 디바이스 함수가 내부 rustc `Instance`로 들어 있습니다. `mir-importer`는 `rustc_public` Stable MIR을 사용하므로 `device_codegen`이 둘을 연결합니다.

```rust
rustc_internal::run(tcx, || {
    let stable_instances: Vec<_> = collected
        .iter()
        .map(|f| rustc_internal::stable(f.instance))
        .collect();
    mir_importer::run_pipeline(&stable_instances, &config)
})
```

1. `rustc_internal::run(tcx, || { ... })`는 `rustc_public`에 필요한 스레드 로컬 컨텍스트를 설정합니다. 범위 밖에서 안정 MIR 쿼리를 하면 패닉합니다.
2. `rustc_internal::stable(f.instance)`는 내부 `rustc_middle::ty::Instance`를 `stable_mir::mir::mono::Instance`로 한 방향 변환합니다. 반대 방향 변환은 지원하지 않습니다.

이후 `mir_importer::run_pipeline()`이 안정 MIR을 읽고 pliron MIR dialect와 PTX까지 전체 lowering을 실행합니다. 자세한 내용은 [MIR importer](mir-importer.md)를 참고하세요.

(rustc-codegen-environment-variables)=

## 환경 변수

`CudaCodegenConfig::from_env()`는 백엔드 동작을 조정하는 선택 환경 변수를 읽습니다. 기본값은 조용한 프로덕션 빌드입니다.

| 변수 | 기능 |
|:--|:--|
| `CUDA_OXIDE_VERBOSE` | 발견 커널, 파이프라인 단계, 시간 등 진행 상황을 출력합니다. |
| `CUDA_OXIDE_DUMP_MIR` | 가져오기 뒤(및 `mem2reg` 뒤) `dialect-mir`를 stderr로 출력합니다. |
| `CUDA_OXIDE_DUMP_LLVM` | lowering 뒤 LLVM dialect 모듈을 stderr로 출력합니다. |
| `CUDA_OXIDE_PTX_DIR` | PTX 출력 디렉터리를 지정합니다. 기본은 호스트 바이너리 옆입니다. |
| `CUDA_OXIDE_TARGET` | GPU 아키텍처를 지정합니다(예: Hopper용 `sm_90a`). |
| `CUDA_OXIDE_SHOW_RUSTC_MIR` | pliron 변환 전 rustc MIR을 출력해 importer 버그를 조사합니다. |

rustc 인자 파서는 코드 생성 백엔드에 정보를 거의 전달하지 않으므로 CLI 플래그보다 환경 변수를 사용합니다. 대상 선택 우선순위는 명시적 `--arch`, `CUDA_OXIDE_TARGET`, `cargo oxide run`의 호스트 CC 자동 감지(빌드/파이프라인 제외), 백엔드 기본값 순입니다.

```bash
CUDA_OXIDE_VERBOSE=1 cargo oxide build <example>
```

verbose 모드는 수집된 함수·크레이트와 각 파이프라인 단계의 시간을 보여줘 컴파일러 디버깅에 유용합니다.

## 전체 흐름

```{figure} images/codegen-flow.svg
:align: center
:width: 100%

코드 생성 백엔드 전체 흐름. CGU를 검색해 커널/디바이스 심볼을 찾고, 발견하면 BFS 수집 후 Stable MIR 브리지를 거쳐 importer가 PTX를 생성합니다. LLVM 호스트 백엔드는 항상 실행됩니다.
```

감싸는 구조 덕분에 나머지 컴파일 파이프라인에서 cuda-oxide는 보이지 않습니다. 링커에는 일반 객체 파일이, 빌드 시스템에는 정상 산출물이 전달됩니다. 호스트 바이너리 옆에 PTX 파일이 생기는 것이 특별한 동작의 흔적이며, 런타임에 CUDA 드라이버가 이를 로드합니다.

:::{seealso}
다음 단계인 [MIR importer](mir-importer.md)는 Stable MIR에서 Pliron IR로의 변환을 설명합니다.
:::
*** End Patch
mkdir -p cuda-oxide/cuda-oxide-book/ko/compiler/images
cp cuda-oxide/cuda-oxide-book/compiler/images/{cross-crate-rlib.svg,codegen-flow.svg} cuda-oxide/cuda-oxide-book/ko/compiler/images/
python - <<'PY'
from pathlib import Path
p=Path('cuda-oxide/cuda-oxide-book/ko/index.md')
s=p.read_text().replace('compiler/pliron\n', 'compiler/pliron\ncompiler/rustc-codegen-cuda\n')
p.write_text(s)
PY
git diff --check && git add cuda-oxide/cuda-oxide-book/ko/compiler/rustc-codegen-cuda.md cuda-oxide/cuda-oxide-book/ko/compiler/images cuda-oxide/cuda-oxide-book/ko/index.md && git commit -m "docs: translate rustc codegen backend chapter" && GIT_ASKPASS=/tmp/cuda-rust-git-askpass GIT_TERMINAL_PROMPT=0 git -c credential.helper= push fork translate/full-ko-book

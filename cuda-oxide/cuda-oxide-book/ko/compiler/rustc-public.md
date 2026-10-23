# rustc_public — 안정 MIR

cuda-oxide는 자체 Rust 파서나 타입 시스템을 만들지 않습니다. 실제 Rust 컴파일러를 활용해 타입 검사와 단형화 이후 `rustc`가 생성하는 내부 표현을 가로채 PTX로 컴파일합니다. 이 장에서는 cuda-oxide가 읽는 중간 표현 MIR, 이를 읽는 안정성 계층 `rustc_public`, 그리고 두 세계를 연결하는 브리지 패턴을 설명합니다.

## MIR이란?

`rustc`는 소스 코드를 파싱하고 이름과 타입을 확인하며 클로저, `for` 루프, `?` 연산자 같은 문법을 낮춘 뒤 **MIR(Mid-level Intermediate Representation, 중간 수준 중간 표현)**을 만듭니다. MIR은 제어 흐름 중심으로 단순화된 프로그램 표현이며 소스의 추상 구문 트리보다 기계 실행에 가까운 형태입니다.

- **빌림 검사:** Rust 소유권 규칙을 소스가 아니라 MIR을 대상으로 검증합니다.
- **최적화:** 상수 전파, 복사 전파, 데드 스토어 제거, 인라이닝이 MIR에서 동작합니다.
- **단형화:** 제네릭 함수의 타입 매개변수를 구체화한 버전을 생성합니다.

일반적인 Rust 컴파일은 MIR을 LLVM IR(또는 `cranelift` 백엔드의 Cranelift IR)로 낮추고 기계어를 생성합니다. cuda-oxide는 MIR이 LLVM에 도달하기 전에 가로채 GPU용 PTX로 컴파일합니다.

## 안정성 문제

MIR은 `rustc`의 **내부** 중간 표현으로 공개 API가 아닙니다. 내부 타입 이름, 열거형 변형 순서, 필드는 나이틀리 릴리스 사이, 때로는 커밋 사이에도 바뀝니다. 컴파일러 팀은 이 내부 구조의 호환성을 보장하지 않습니다.

cuda-oxide가 `rustc_middle` 타입을 직접 사용한다면 나이틀리 업데이트마다 내부 변경을 따라가야 합니다. GPU 코드를 쓰는 대신 컴파일 오류를 패치하는 데 시간을 쓰게 됩니다. 이 문제를 완화하는 계층이 `rustc_public`입니다.

## rustc_public이란?

이전에 `stable_mir`로 불린 `rustc_public`은 Rust 컴파일러 내부에 접근하는 **안정 인터페이스**입니다. 검증기, 린터, cuda-oxide 같은 코드 생성 백엔드가 컴파일러 구현 변경에 매번 깨지지 않고 분석과 코드 생성을 수행하도록 합니다.

구현은 rustc 저장소의 두 크레이트에 있습니다.

| 크레이트 | 역할 |
|:--|:--|
| `rustc_public` | 사용자용 공개 API입니다. MIR의 `Body`, `BasicBlock`, `Local`, `Place`, `Ty`, `StatementKind`, `TerminatorKind` 등의 안정 타입을 정의하며 향후 crates.io에 배포될 예정입니다. |
| `rustc_public_bridge` | 공개 타입과 컴파일러 내부 `rustc_middle` 타입 사이를 변환합니다. |

안정 API에서 주로 사용하는 타입은 다음과 같습니다.

- **`Body`** — 단일 함수의 MIR(기본 블록, 로컬, 타입).
- **`BasicBlock`** — 문장들을 순서대로 실행하고 종결자로 끝나는 블록.
- **`Local`, `Place`** — 변수와 메모리 위치.
- **`Ty`** — 기본 타입, 참조, 튜플, ADT, 함수 포인터, 클로저를 포함한 Rust 타입.
- **`StatementKind`** — 대입, 저장소 주석, 판별값 읽기.
- **`TerminatorKind`** — 분기, 호출, 반환, assert, drop.
- **`Instance`** — 타입 인자가 구체화된 단형화 함수.

```{note}
`rustc_public`은 Rust 정형 검증을 위한 AWS의 [Kani](https://github.com/model-checking/kani) 팀과 안정적인 컴파일러 접근이 필요한 여러 프로젝트가 주도합니다. cuda-oxide는 브리지를 직접 유지하지 않고 이 작업의 혜택을 받습니다.
```

## cuda-oxide 연결 방식

### CodegenBackend 트레이트

`rustc`의 컴파일은 `CodegenBackend` 트레이트를 중심으로 구성됩니다. 핵심 메서드 `codegen_crate`는 타입 정보, MIR 본문, 메타데이터를 담은 컴파일러 객체 `TyCtxt`를 받고 컴파일 결과를 만들어야 합니다.

일반적으로 `rustc_codegen_llvm`이 이 트레이트를 구현해 MIR을 LLVM으로 기계어로 바꿉니다. cuda-oxide는 LLVM을 대체하지 않고 `CudaCodegenBackend`로 감쌉니다. GPU 측 코드를 처리하고 호스트 코드는 LLVM에 맡기는 설계입니다.

1. `rustc`가 `CudaCodegenBackend::codegen_crate(tcx)`를 호출합니다.
2. cuda-oxide가 가로채 collector를 실행해 커널과 전이 호출 대상을 포함한 모든 디바이스 함수를 찾습니다. 안정 MIR 컨텍스트에 진입해 `mir-importer`를 통해 PTX로 컴파일합니다.
3. PTX 산출물을 빌드 산출물과 함께 디스크에 기록합니다.
4. cuda-oxide가 호스트 코드를 감싼 LLVM 백엔드에 전달하고 LLVM이 평소처럼 네이티브 바이너리를 생성합니다.

이 구조로 cargo 호출 한 번에 네이티브 호스트 바이너리와 PTX 모듈을 모두 만들 수 있어, 별도의 도구 체인이나 분리된 빌드 시스템이 필요하지 않습니다.

### 안정 MIR 컨텍스트 진입

`codegen_crate` 안에서 cuda-oxide는 불안정한 내부 타입인 `rustc_middle`을 받습니다. MIR을 Pliron IR로 변환하는 `mir-importer`는 주 인터페이스로 `rustc_public`을 사용합니다. 둘 사이를 연결하는 예시는 다음과 같습니다.

```rust
// codegen_crate() 내부:
let result = rustc_internal::run(tcx, || {
    // 이제 안정 MIR 컨텍스트 안입니다.
    let stable_instance = rustc_internal::stable(func.instance);
    let body = stable_instance.body().unwrap();
    // cuda-oxide 파이프라인에 전달합니다.
    mir_importer::run_pipeline(&functions, &config)
});
```

`rustc_internal::run(tcx, || { ... })`는 안정 MIR 쿼리를 사용할 수 있는 범위 지정 컨텍스트를 설정합니다. `rustc_internal::stable()`은 내부 `rustc_middle::ty::Instance<'tcx>`를 안정 `rustc_public::mir::mono::Instance`로 변환합니다. 이후 `Instance::body()`는 안정 API를 통해 MIR을 반환하므로 `rustc_middle`을 직접 다루지 않아도 됩니다.

## 스레드 로컬 컨텍스트 관리

브리지가 컨텍스트를 함수 인자로 넘기지 않고 특별한 `run()` 범위를 요구하는 이유는 수명 얽힘 때문입니다. `TyCtxt<'tcx>`는 컴파일러 arena의 데이터를 빌립니다. 구조체에 저장하거나 함수 사이에 전달할 수 있지만 `'tcx` 수명은 컴파일러 세션 안에 있어야 합니다. `rustc_public`은 매 호출마다 `TyCtxt` 인자를 요구하지 않도록 범위가 지정된 스레드 로컬 저장소에서 현재 컨텍스트를 찾습니다.

브리지는 스레드 로컬 변수(TLV)를 설정합니다.

| TLV | 저장값 | 용도 |
|:--|:--|:--|
| `compiler_interface::TLV` | `Cell<*const ()>` | 현재 `CompilerInterface` 참조에 범위 내에서 접근합니다. |

내부적으로 포인터 타입을 지운 뒤 `compiler_interface::with()`가 복원해 `local_crate()`, `all_local_items()` 등의 쿼리에 사용합니다. `rustc_internal::stable(instance)` 같은 변환은 `with_bridge()`를 거쳐 같은 인터페이스와 변환 테이블을 이용합니다. 별도의 두 번째 스레드 로컬 컨텍스트는 없습니다.

인스턴스를 변환한 다음 MIR 본문을 요청하는 과정은 컨텍스트 하나를 계속 사용합니다.

```text
rustc_internal::run(tcx, || {
    stable_instance = stable(instance) -> with_bridge() -> with() -> tables
    stable_instance.body() -----------------------------> with() -> query
})
```

활성 범위 밖의 쿼리는 패닉하고, 중첩 `run()`은 오류를 반환합니다. 현재 설계는 [rust-lang/rust#147923](https://github.com/rust-lang/rust/pull/147923)에서 예전 `Container` 및 이중 컨텍스트 구현을 대체했습니다.

## 브리지 패턴

`CompilerInterface`는 rustc 내부 ID와 안정 API 타입을 양방향으로 연결하는 **`Tables`** 구조체를 소유합니다. `rustc_internal::stable(instance)`를 호출하면 브리지가 테이블에서 안정 ID를 찾거나 새로 만듭니다. 안정 API가 함수 MIR 본문처럼 컴파일러 쿼리를 요청할 때는 반대 방향으로 테이블을 조회해 내부 타입을 복원합니다.

```text
    rustc_middle::ty::Instance<'tcx>
              │
              ▼
         ┌─────────┐
         │  Tables │  (양방향: 내부 ↔ 안정)
         └─────────┘
              │
              ▼
    rustc_public::mir::mono::Instance
```

- **내부 가변성:** `CompilerInterface`는 테이블과 컴파일러 컨텍스트를 `RefCell`에 보관합니다. 쿼리는 현재 스레드에서 빌리며 겹치는 가변 빌림은 런타임에 거부됩니다.
- **캐시:** 변환된 타입이나 인스턴스는 테이블에 저장되어 이후 조회에서 재사용됩니다.
- **자동 정리:** `rustc_internal::run()`이 반환되면 범위의 스레드 로컬 값이 해제되고 인터페이스와 테이블도 삭제됩니다. 안정 ID는 이 테이블에 종속되므로 다음 컨텍스트에서 재사용하면 안 됩니다.

`rustc-codegen-cuda`의 드라이버는 `rustc_public::rustc_internal::run()`으로 범위를 열고 `stable()`로 인스턴스를 변환합니다. `mir-importer`는 대부분 `rustc_public` 타입을 사용하지만 일부 검사는 브리지를 통해 `rustc_middle`을 확인합니다. 예를 들어 grid-constant 검사는 참조 수명과 인라이닝 소유권을 읽습니다. importer가 브리지 테이블 자체를 관리하지는 않습니다.

## MIR 형태

cuda-oxide가 MIR을 변환하기 전에 MIR의 형태를 살펴봅시다. 간단한 함수와 그 MIR입니다.

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

```text
// `add`의 MIR:
// _0: i32              (반환 위치)
// _1: i32              (인자 `x`)
// _2: i32              (인자 `y`)
// _3: (i32, bool)      (검사 산술 임시값)
//
// bb0: {
//     _3 = CheckedAdd(_1, _2);
//     assert(!(_3.1), "attempt to compute `{} + {}`, which would overflow") -> bb1;
// }
// bb1: {
//     _0 = (_3.0);
//     return;
// }
```

- **로컬**은 번호가 붙습니다. `_0`은 결과가 저장되는 반환 위치이며 `_1`, `_2` 등은 함수 인자입니다. 번호가 더 큰 로컬은 컴파일러가 만든 임시값입니다.
- **기본 블록**(`bb0`, `bb1` 등)은 문장들의 직선 실행 구간입니다. 각 블록은 분기, 호출, 반환, assert 등 제어를 넘기는 **종결자** 하나로 끝납니다.
- **`CheckedAdd`**는 `(i32, bool)` 튜플을 반환합니다. `bool`은 오버플로 플래그이며 assert 종결자가 확인해 참이면 다음 블록으로, 아니면 패닉 경로로 갑니다. 디버그 빌드에서 정수 오버플로를 잡으며 릴리스 빌드에서는 검사가 최적화될 수 있습니다.
- **중첩 표현식이 없습니다.** 소스의 `x + y`는 검사 덧셈 계산과 결과 추출이라는 별도 연산이 됩니다. 각 중간값에는 로컬이 할당됩니다. 이 평탄한 구조 덕분에 도구가 MIR을 처리하기 쉽습니다.

```{note}
`rustc`에 `--emit=mir`를 전달하거나 [Rust Playground](https://play.rust-lang.org/)에서 MIR 출력을 선택하면 함수 MIR을 볼 수 있습니다. 로컬 번호에 익숙해지면 상당히 읽기 쉽습니다.
```

### cuda-oxide에서 중요한 이유

MIR의 평탄하고 명시적인 구조가 이를 기반으로 GPU 컴파일러를 만들 수 있게 합니다. AST나 HIR(고수준 IR)을 직접 처리한다면 클로저, 메서드 해석, 트레이트 디스패치, 타입 추론 등 MIR 생성 시점에 이미 해결된 언어 기능을 다시 다뤄야 합니다. MIR을 읽으면 제네릭은 단형화되고 클로저는 구조체로 낮춰지며 제어 흐름도 명시되어 있습니다. `mir-importer`가 이를 Pliron IR(MLIR 유사 프레임워크)로 변환하고 [낮춤 파이프라인](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/lowering-pipeline.html)이 PTX까지 처리합니다.

## 나이틀리 버전 고정

`rustc_public`이 안정 API를 제공해도 브리지(`rustc_public_bridge`)는 컴파일러 내부 타입에 맞춰 빌드되며 독립적으로 버전이 지정되지는 않습니다. 따라서 특정 cuda-oxide 버전은 특정 나이틀리 버전과 함께 동작합니다.

cuda-oxide는 `rust-toolchain.toml`에서 정확한 나이틀리 버전을 고정합니다.

```toml
[toolchain]
channel = "nightly-2026-10-11"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
```

이 고정으로 저장소를 복제한 모든 사용자가 같은 컴파일러, MIR 형태, 브리지 동작을 사용합니다. 버전을 갱신할 때는 다음 절차를 따릅니다.

1. `cuda-oxide/rust-toolchain.toml`의 날짜를 올립니다. rustup은 독립 `rustc-codegen-cuda` 워크스페이스에서 상위 경로를 찾아 이 설정을 사용하므로 중첩 사본은 필요하지 않습니다. `cuda-oxide/scripts/check-toolchain-parity.sh`로 스캐폴드, devcontainer, 책과 README의 날짜가 지정된 명령을 검사합니다.
2. `rustc_public` API 변경을 반영합니다. 안정 API라면 대개 변경은 작습니다.
3. 전체 테스트를 실행해 모든 예제가 계속 컴파일되고 올바른 PTX를 만드는지 확인합니다.
4. 문제가 있으면 되돌리고 다음 나이틀리를 시도합니다.

```{note}
`rustc_public`이 성숙해 crates.io 배포에 가까워지면 특정 나이틀리에 대한 결합은 약해질 것입니다. 장기 목표는 충분히 최신인 안정 Rust 도구 체인에서도 cuda-oxide가 동작하게 하는 것이지만, 아직 그 단계는 아닙니다.
```

cuda-oxide가 Rust 컴파일러와 통신하는 방식을 살펴봤습니다. 다음 장에서는 이 과정이 코드 생성 백엔드에 도달했을 때 일어나는 일, 즉 [코드 생성기](rustc-codegen-cuda.md)를 다룹니다.

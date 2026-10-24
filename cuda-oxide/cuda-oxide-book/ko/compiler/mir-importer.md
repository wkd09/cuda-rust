# MIR 임포터

앞 장에서는 rustc가 Stable MIR([rustc_public](rustc-public.md))을 생성하는 방식과 pliron이 IR 프레임워크([Pliron](pliron.md))를 제공하는 방식을 설명했습니다. 이 장에서는 두 가지가 만납니다. `mir-importer`는 rustc가 전달한 Stable MIR을 Rust 의미 체계를 보존하는 pliron dialect인 `dialect-mir`로 변환합니다. 변환기는 먼저 alloca/load/store 형식을 출력합니다. 생성 비용이 낮고 이해하기 쉬우며 입력과 pliron 식별성을 보존합니다. 이후 `pliron::opts::mem2reg` 패스가 슬롯을 SSA 형식으로 승격합니다. 그런 다음 cuda-oxide는 주석이 붙은 루프 언롤링부터 시작해 [컴파일러 최적화](compiler-optimizations.md)를 수행한 뒤 LLVM dialect로 하위 변환합니다.

변환은 작업의 절반일 뿐입니다. `mir-importer`는 `dialect-mir` 변환, 검증, 최적화, 하위 변환, 내보내기, PTX 생성이라는 *전체* 컴파일 파이프라인도 조율합니다. 변환기이자 무대 감독입니다.

크레이트는 `cuda-oxide/crates/mir-importer`에 있으며 두 부분으로 구성됩니다.

- **`translator/`** — MIR에서 pliron으로 변환하는 로직(핵심 부분).
- **`pipeline.rs`** — 모든 단계를 연결하는 오케스트레이션(전체 진행 담당).

---

## 파이프라인 오케스트레이션

세부 변환 내용을 살펴보기 전에 전체 흐름부터 보겠습니다. `rustc-codegen-cuda`는 디바이스 함수를 수집한 뒤 진입점인 `run_pipeline()`을 호출합니다. 이 함수는 `CollectedFunction` 구조체 목록과 `PipelineConfig`를 받아 다음 단계를 실행합니다.

```text
Step 1:  Translate Rust MIR → `dialect-mir`
Step 2:  Verify `dialect-mir` module
Step 3:  Run `pliron::opts::mem2reg` to promote alloca slots back into SSA
Step 4:  Apply `dialect-mir` optimizations (currently annotated loop unrolling)
Step 5:  Lower `dialect-mir` → LLVM dialect (via mir-lower)
Step 6:  For NVVM builds, adjust the LLVM dialect via nvvm-transforms
Step 7:  Export the LLVM dialect to textual LLVM IR (.ll)
Step 8:  Compile with llc, or with libNVVM and nvJitLink
```

전체 변수 디버그 빌드에서는 3단계와 4단계를 건너뛰어 소스 변수가 cuda-gdb가 접근할 수 있는 안정된 메모리 위치에 남도록 합니다.

각 `CollectedFunction`에는 디바이스 함수에 대한 파이프라인 정보가 모두 담깁니다.

```rust
pub struct CollectedFunction {
    pub instance: Instance,
    pub is_kernel: bool,
    pub export_name: String,
}
```

`instance`는 `rustc_public`이 제공하는 단형화된 함수입니다. `is_kernel`은 커널 진입점과 디바이스 헬퍼 함수를 구분합니다(커널은 LLVM IR에 특수 메타데이터를 받아 NVPTX 백엔드가 `.entry` 지점으로 출력합니다). `export_name`은 최종 PTX에 표시되는 심볼 이름입니다. 디바이스 함수의 경우 보통 같은 함수에 대해 `CrateDef::name()`이 반환하는 정규화된 전체 이름(FQDN)입니다.

각 함수에 대해 파이프라인은 다음을 수행합니다.

1. `instance.body()`로 MIR 본문을 가져옵니다.
2. `translator::body::translate_body()`를 호출해 함수의 `dialect-mir` 표현을 생성합니다(지역 변수에는 `mir.alloca` 슬롯 사용). 이후 전체 실행이 공유하는 `builtin.module`에 추가합니다.
3. 타입 불일치, 피연산자 누락, dominance 위반 같은 구조 오류가 이후 난해한 LLVM 실패로 번지기 전에 pliron 검증기로 모듈을 검사합니다.
4. `pliron::opts::mem2reg`를 실행해 alloca 슬롯을 `dialect-mir`의 SSA 값으로 승격합니다.
5. [컴파일러 최적화](compiler-optimizations.md)를 실행합니다. 현재는 `#[unroll]` 또는 `#[unroll(N)]` 요청이 있는 지원 루프를 언롤합니다.
6. `mir-lower` 크레이트의 `lower_mir_to_llvm`을 실행해 `DialectConversion`으로 모든 `dialect-mir` 연산을 LLVM dialect 대응 연산으로 하위 변환합니다.
7. NVVM 빌드에서는 대상 GPU에 맞는 dialect로 `nvvm-transforms`를 실행합니다. 일반 PTX 빌드는 이 단계를 건너뜁니다.
8. LLVM dialect 모듈을 텍스트 `.ll` 문자열로 내보내 `llc`로 컴파일합니다. NVVM 출력은 libNVVM 및 nvJitLink로 컴파일합니다.

어느 단계에서든 실패하면 파이프라인이 중단되고 문제를 진단할 충분한 맥락을 담은 타입 지정 오류(`NoBody`, `Translation`, `Verification`, `Lowering`, `Export`, `PtxGeneration`)를 반환합니다. 조용한 손상이나 원인 모를 빈 출력 파일은 없습니다.

---

## 변환기 아키텍처

Stable MIR은 `translator/` 디렉터리에서 pliron IR로 변환됩니다. 각 모듈은 MIR 구조의 한 수준을 담당하며 다음처럼 조합됩니다.

| 모듈       | 역할                                                                    |
| :----------- | :--------------------------------------------------------------------------|
| `body`       | 함수 수준 변환, alloca 슬롯 준비, FQDN 이름 정리    |
| `block`      | 기본 블록 변환 조정                                        |
| `statement`  | 문장 변환(대입, 저장 공간)                              |
| `terminator` | 종단자 변환(goto, call, return, FQDN 기반 내장 함수 디스패치) |
| `rvalue`     | 표현식 변환(이항 연산, 캐스트, 집계 값)                         |
| `types`      | Rust 타입을 `dialect-mir` 타입으로 변환                                 |
| `values`     | MIR 지역 변수 → alloca 슬롯 매핑(`ValueMap`) 및 슬롯 주소 공간 추론    |

호출 흐름은 MIR 구조를 따라 위에서 아래로 진행됩니다.

```text
pipeline::run_pipeline()
  └─ body::translate_body()               // once per collected function
       ├─ emit_entry_allocas()            // one mir.alloca per non-ZST local
       │     └─ SlotAddrSpaceMap::analyze // pointer slot addrspace inference
       └─ For each basic block:
            └─ block::translate_block()
                  ├─ statement::translate_statement()
                  │     └─ rvalue::translate_rvalue()
                  └─ terminator::translate_terminator()
```

`translate_body()`는 함수 시그니처를 설정하고 MIR 기본 블록에 대응하는 pliron 블록을 만듭니다. 함수 이름을 정리하고(`instance.name()`의 FQDN에서 `::`을 `__`로 변경), 진입 블록 맨 앞에 ZST가 아닌 각 지역 변수마다 `mir.alloca`를 출력한 뒤 입력 함수 인자를 해당 슬롯에 저장합니다. 진입점이 아닌 블록에는 인자를 두지 않습니다. 블록 간 데이터 흐름은 블록 인자가 아니라 alloca 슬롯을 통합니다. 그런 다음 각 블록을 순서대로 순회하며 문장과 종단자를 하나씩 변환합니다.

`translate_statement()`은 대입, 저장 공간 live/dead 표시, 판별값 쓰기 등 블록 안의 단순 연산을 처리합니다. 대입 오른쪽에 MIR `Rvalue` 표현식이 있으면 `translate_rvalue()`에 맡깁니다. 이 함수는 이항/단항 연산, 캐스트, 집계 값 생성, 판별값 읽기, 포인터 산술 등 Rust가 컴파일하는 여러 항목을 처리합니다.

`translate_terminator()`는 `Goto`, `SwitchInt`, `Call`, `Return`, `Assert`, `Drop`, `Unreachable` 같은 블록 종료 연산을 처리합니다. 내장 함수 디스패치도 여기서 수행하며 아래에서 따로 설명합니다.

---

## SSA 과제와 처리 시점

변환에서 가장 까다로운 부분이므로 자세히 설명하겠습니다.

Rust MIR은 엄격한 SSA 형식이 *아닙니다*. 지역 변수는 이름이 있는 저장 위치이며 모든 블록에서 읽거나 쓸 수 있습니다. `_3`에 `bb0`에서 값을 대입하면 `bb1`, `bb5` 또는 다른 블록 어디서든 자유롭게 사용할 수 있습니다. MIR은 이를 제한하지 않습니다.

Pliron IR(MLIR 유사)은 최종적으로 **엄격한 SSA**를 요구합니다. 값은 모든 사용을 지배해야 하며 블록 사이로 값을 전달하려면 **블록 인자**로 명시적으로 넘겨야 합니다. 다른 블록의 지역 변수에 임의로 접근할 수 없습니다.

이 차이는 두 단계로 해결합니다.

1. **임포터: alloca + load/store.** `mir-importer`는 의도적으로 SSA를 직접 만들지 *않습니다*. ZST가 아닌 모든 MIR 지역 변수에는 스택 슬롯 하나가 대응합니다. 진입 블록 시작 부분에 `mir.alloca`를 출력하고 `ValueMap`에 기록합니다. 지역 변수 쓰기는 슬롯에 `mir.store`, 읽기는 `mir.load`가 됩니다. 따라서 분기 종단자는 모두 **피연산자가 없고** 진입점이 아닌 블록은 **인자가 없습니다**. 블록 간 데이터 흐름은 블록 인자가 아니라 alloca 슬롯을 통합니다.

2. **`pliron::opts::mem2reg`: 슬롯에서 SSA로.** `dialect-mir` 모듈 검증 후 `pipeline.rs`가 pliron 내장 `mem2reg` 패스를 실행합니다. 승격 가능한 모든 alloca를 다시 SSA 값으로 바꾸고 각 load를 도달하는 정의에 연결합니다. 여러 제어 경로에서 값이 합쳐지는 곳에는 블록 인자(pliron의 phi 노드 표현)를 삽입합니다. 실제로 스택에 남겨야 하는 주소 취득 슬롯은 그대로 두며 `dialect-mir`에서 LLVM dialect로 낮출 때 실제 `alloca`로 변환합니다.

문제를 간단한 예로 보겠습니다. `_1`을 `bb0`에서 쓰고 `bb1`에서 읽는 MIR은 다음과 같습니다.

```text
// Rust MIR
bb0: { _1 = 42_i32; goto -> bb1; }
bb1: { _0 = _1;     return; }
```

임포터는 먼저 다음과 같은 alloca 기반 `dialect-mir`를 출력합니다(`mem2reg` 실행 전).

```text
^bb0:
  %s1 = mir.alloca           : !mir.ptr<i32>
  %c  = mir.constant 42_i32  : i32
  mir.store %c, %s1
  mir.goto ^bb1                     // zero-operand; _1 flows via %s1
^bb1:                               // no block arguments
  %r = mir.load %s1 : i32
  mir.return %r : i32
```

`pliron::opts::mem2reg`가 `%s1`을 승격하면 같은 함수가 SSA 형식의 `dialect-mir`가 됩니다. 도달 정의를 병합해야 하는 위치에만 블록 인자가 나타납니다.

```text
^bb0:
  %c = mir.constant 42_i32 : i32
  mir.goto ^bb1(%c : i32)
^bb1(%r : i32):
  mir.return %r : i32
```

(위 예처럼 선행 블록이 하나인 후속 블록에는 블록 인자 하나가 생깁니다. 여러 도달 정의가 합쳐지는 지점에서는 `mem2reg`가 phi 형태의 인자를 추가합니다.)

요약하면 임포터는 생존 분석을 수행하지 않고 블록 사이로 값을 전달하지 않습니다. “어떤 값이 어디서 살아 있는가?”라는 추론은 `mem2reg`에 맡깁니다. 이 패스는 전체 `dialect-mir` 모듈에 대해 한 번에 올바르게 처리합니다. 빠르게 참고할 수 있도록 `translator/terminator/mod.rs` 모듈 문서에 같은 예제를 인라인 소스 형식으로 실었습니다.

---

## 타입 변환

`types` 모듈은 `rustc_public`에서 얻은 Rust 타입을 `dialect-mir` 타입으로 변환합니다. 대부분 대응은 단순하지만 몇 가지는 주의가 필요합니다.

| Rust 타입             | `dialect-mir` 타입            | 설명                                |
| :-------------------- | :---------------------------- | :----------------------------------- |
| `i32`, `u64` 등    | `IntegerType`                 | 부호 정보 포함             |
| `f32`, `f64`          | `Float32Type` / `Float64Type` | 표준 IEEE 754                    |
| `bool`                | `IntegerType(1)`              | 관례에 따라 1비트 정수       |
| `(A, B, C)`           | `MirTupleType`                | 서로 다른 타입의 곱 타입           |
| `&[T]`                | `MirSliceType`                | 포인터 + 길이                     |
| `DisjointSlice<T>`    | `MirDisjointSliceType`        | 범위 검사 및 타입 인덱스가 적용된 가변 슬라이스 |
| `struct Foo`          | `MirStructType`               | rustc 레이아웃의 필드 오프셋 포함 |
| `*mut T` / `*const T` | `MirPtrType`                  | GPU 주소 공간 포함               |
| `enum Option<T>`      | `MirEnumType`                 | 판별값 + variant              |

### 동적 구조체 레이아웃

사용자가 겪을 수 있는 골칫거리를 없애는 미묘한 부분을 살펴보겠습니다. 다음 구조체를 보세요.

```rust
struct Extreme {
    a: u8,
    b: i128,
}
```

Rust 레이아웃 알고리즘은 정렬을 위해 필드 순서를 바꿀 수 있습니다.

```text
User writes:      struct Extreme { a: u8, b: i128 }
rustc may layout: [b: i128 @ offset 0][a: u8 @ offset 16]
MirStructType:    mem_to_decl mapping, offsets, total_size
LLVM struct:      { i128, i8, [15 x i8] }   // explicit padding
```

cuda-oxide는 rustc에서 모든 필드의 정확한 바이트 오프셋을 조회하고 명시적 패딩 바이트를 포함한 구조체 타입을 만듭니다. `MirStructType`에는 메모리 순서에서 선언 순서로 가는 `mem_to_decl` 매핑, 필드별 오프셋, 전체 크기가 저장됩니다. LLVM으로 하위 변환할 때 필드 사이의 패딩은 `[N x i8]` 배열로 실체화됩니다.

실용적인 결론은 호스트와 디바이스 코드에서 공유하는 타입에 **`#[repr(C)]가 필요하지 않다**는 것입니다. cuda-oxide가 rustc 레이아웃과 자동으로 일치시키므로 구조체는 Rust 기본 `repr(Rust)` 레이아웃을 사용할 수 있고 컴파일러가 양쪽에서 올바르게 처리합니다. 기억할 속성이 하나 줄고 실수할 가능성도 줄어듭니다.

---

## 내장 함수 디스패치

변환기가 `Call` 종단자를 만나도 즉시 `mir.call`을 출력하지 않습니다. 먼저 호출 대상이 *알려진 내장 함수*인지 확인합니다. 이는 본문이 있는 일반 함수가 아니라 GPU 하드웨어 명령에 직접 대응하는 `cuda_device` 함수입니다.

`try_dispatch_intrinsic()` 함수는 `CrateDef::name()`에서 얻은 호출 대상의 **정규화된 전체 이름(FQDN)**을 비교합니다.

```rust
match name {
    // condensed; the real arms also match re-export and ABI-alias paths
    "cuda_device::thread::threadIdx_x"  => emit_generated_nvvm_intrinsic(ReadPtxSregTidXOp),
    "cuda_device::thread::sync_threads" => { /* builds a Barrier0Op in place */ }
    // ... 100+ intrinsics
    _ => translate_as_normal_call()
}
```

대부분의 분기는 수작업으로 작성하지 않습니다. `intrinsics/catalog.json`에 기술된 항목은 `cuda-oxide/crates/mir-importer/src/translator/terminator/intrinsics/generated/` 아래의 유형별 모듈에 자동 생성됩니다. 위의 두 분기와 워프 셔플도 여기에 해당합니다. 비교 문자열은 마스크가 있는 `_sync` 형식(`shuffle_xor_sync`, `shuffle_xor_f32_sync`, `shuffle_xor_u64_sync`, `shuffle_up_*`/`shuffle_down_*` 계열)입니다. `warp::shuffle_xor` 자체는 일치 문자열이 아닙니다. `cuda-oxide/crates/cuda-device/src/warp.rs`의 `#[inline(always)]` 래퍼로 전체 워프 마스크(`u32::MAX`)를 제공해 `shuffle_xor_sync`를 호출하므로 디스패처는 `_sync` 호출만 가로챕니다. `terminator/mod.rs`의 수작업 분기와 생성 파일 옆 `intrinsics/` 모듈에는 카탈로그가 설명하지 않는 경우가 있습니다.

서로 다른 모듈에 같은 이름의 함수가 있을 때 혼동하지 않도록 `threadIdx_x`만이 아니라 전체 FQDN(예: `cuda_device::thread::threadIdx_x`)을 비교에 사용합니다. 제네릭이 아니며 내장 함수도 아닌 호출의 대상 이름에도 같은 FQDN을 사용합니다. 수집기가 일치하는 이름을 생성하고 하위 변환 계층이 양쪽의 `::`을 `__`로 바꿉니다.

함수가 인식된 내장 함수이면 변환기는 대응하는 `dialect-nvvm` 연산을 직접 출력합니다. 함수 본문이나 호출 오버헤드 없이 하드웨어 명령으로 연결됩니다. 스레드 인덱싱, 워프 셔플, 배리어, 공유 메모리 연산, TMA 대량 복사, 행렬 곱셈 명령이 모두 이 경로를 거칩니다.

함수가 내장 함수가 *아니면* 일반 경로로 진행해 심볼 이름으로 호출 대상을 참조하는 `mir.call` 연산을 출력합니다. 호출 대상 함수 본문은 별도로 변환되어 있습니다(수집된 함수 목록에도 포함됨). 따라서 정상적으로 링크됩니다.

```{note}
내장 함수 디스패치 테이블은 cuda-oxide에 새 GPU 연산을 추가하는 주요 확장 지점입니다. NVIDIA가 새 명령을 제공하고 이를 노출하려면 `cuda_device` 함수, `dialect-nvvm` 연산, 여기의 비교 분기를 추가하세요. 단계별 설명은 [새 내장 함수 추가](adding-new-intrinsics)를 참조하세요.
```

---

## 언와인드 경로 처리

MIR은 Rust 패닉 의미 체계를 충실히 모델링합니다. 모든 함수 호출에는 반환 대상과 언와인드 대상, 두 후속 경로가 있습니다.

```text
_2 = mul(_1, _3) -> [return: bb1, unwind: bb2]
```

CPU에서는 언와인드 경로가 중요합니다. 소멸자를 실행하고 스택을 풀어 패닉을 잡거나 프로세스를 중단합니다. GPU에서는 현재 CUDA 툴체인이 이 기능을 제공하지 않습니다. nvcc/ptxas가 landing pad를 제거하며 PTX에는 예외 처리 기반이 남지 않습니다. 하드웨어 자체는 언와인드를 지원할 *수도* 있습니다(Volta 이후 절대 분기와 스레드별 호출 스택 추적이면 충분합니다). 하지만 컴파일러와 런타임은 이를 연결하지 않았습니다. NVIDIA는 자동차 안전을 위해 C++ 예외 지원을 추가하는 프로젝트를 진행 중이며 현재 cuda-oxide 설계는 향후 해당 작업과 호환됩니다.

현재 cuda-oxide는 **모든 언와인드 경로를 도달 불가로 처리합니다**. 런타임에 패닉이 발생하면(예: 디버그 모드 정수 오버플로나 명시적 `panic!()`) GPU 트랩이 발생하고 커널이 충돌합니다. 사용자가 플래그를 설정하지 않아도 의미상 `panic=abort`와 같습니다.

실제로 변환기는 모든 `Call` 및 `Assert` 종단자의 언와인드 대상을 무시하고 반환 경로 분기만 생성합니다. 언와인드 블록은 변환되지 않으므로 처음부터 없었던 것처럼 사라집니다.

패닉 *메시지*도 함께 사라집니다. `core::panicking`으로 발산 호출하는 종단자가 있는 블록은 `nvvm.trap`과 `mir.unreachable`로 낮아질 뿐입니다. 블록의 문장은 제거된 호출이 사용할 메시지 `&str`, `format_args!` 조각을 만드는 데만 쓰이며 후속 블록이 없으므로 이 문장들이 쓰는 값은 읽히지 않습니다. 이런 문장을 건너뛰기 때문에 메시지를 포함한 패닉도 컴파일할 수 있습니다. 실체화된 `&str` 상수는 디바이스 하위 변환을 지원하지 않습니다.

생각만큼 위험한 상황은 아닙니다. Rust의 빌림 검사기와 타입 시스템은 패닉을 유발할 오류 대부분을 막습니다. 배열 범위 초과나 `None`에 대한 unwrap처럼 남은 오류는 GPU 트랩이 올바른 동작입니다. 커널 실행 중 논리 오류가 발생했을 때 GPU 스레드가 유용하게 “복구”할 방법은 없습니다.

Drop glue에도 같은 규칙을 적용합니다. 임포터는 소멸자가 관찰 가능한 작업 없이 반환한다는 증명이 있을 때만 소멸자 호출을 지웁니다. 무한 반복하는 소멸자나 이미 증명 중인 인스턴스를 다시 호출하는 소멸자는 호출을 유지하므로 Rust 소스와 마찬가지로 그 뒤의 코드는 실행되지 않습니다.

```{figure} images/drop-proof-termination.svg
:align: center
:width: 100%

Drop 무연산 증명에는 소멸자가 반환한다는 조건이 필요합니다. 이전에는 `loop {}` 소멸자 호출이 제거되어 뒤의 저장이 실행됐습니다. 이제 호출이 유지되므로 PTX는 한 번 저장한 뒤 반복합니다. 반면 합류 지점으로 돌아오는 `if`/`else` 소멸자는 여전히 제거됩니다.
```

---

## 전체 흐름 정리

각 부분이 어떻게 연결되는지 간단한 커널을 전체 파이프라인에 따라 추적해 보겠습니다. 다음은 벡터 덧셈 커널입니다.

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

`mir-importer`가 Stable MIR을 `dialect-mir`로 변환하고 `pliron::opts::mem2reg`가 alloca 슬롯을 SSA로 승격하면 다음과 같은 결과가 나옵니다(간결성을 위해 세부 사항은 생략).

```text
mir.func @vecadd(%a: mir.slice<f32>, %b: mir.slice<f32>,
                 %c: mir.disjoint_slice<f32>) {
^entry:
    %idx = nvvm.read_ptx_sreg_tid_x : i32
    %len = mir.extract_field %c[1]       // slice length
    %in_bounds = mir.lt %idx, %len
    mir.cond_br %in_bounds, ^compute, ^exit

^compute:
    %a_val = mir.load ...                // a[idx]
    %b_val = mir.load ...                // b[idx]
    %sum = mir.add %a_val, %b_val : f32
    mir.store %sum, ...                  // c[idx] = sum
    mir.goto ^exit

^exit:
    mir.return
}
```

몇 가지 살펴볼 점이 있습니다.

- **`thread::index_1d()`**는 내장 함수로 디스패치되어 `nvvm.read_ptx_sreg_tid_x`가 됩니다. 함수 호출이 아니라 GPU 레지스터를 직접 읽습니다.
- **`DisjointSlice::get_mut()`**은 범위 검사(`mir.lt`)와 조건부 분기로 바뀝니다. Rust의 `if let Some` 패턴이 명시적 제어 흐름이 됩니다.
- **블록 인자**는 여기서 보이지 않습니다. `mir-importer`는 모든 분기 종단자를 피연산자 없이, 진입점이 아닌 모든 블록을 인자 없이 출력합니다. `pliron::opts::mem2reg`가 실행되기 전까지 값은 alloca 슬롯을 통해 블록 경계를 넘습니다. 이 함수에서는 `^compute`나 `^exit`에서 병합할 값이 없으므로(`mem2reg`가 슬롯을 승격하면서) 블록 인자를 추가하지 않습니다. 루프 뒤쪽까지 값이 살아 있는 커널이라면 `mem2reg` 후 `mir.goto ^header(%i, %acc)`와 `^header(%i: i32, %acc: f32)`가 만들어집니다. 이는 pliron에서 phi 노드를 나타내는 형식입니다.
- **언와인드 경로 없음.** 원래 MIR에는 패닉이 가능한 모든 연산에 언와인드 대상이 있었지만 여기서는 제거되었습니다.

이후에는 파이프라인이 이어받습니다.

1. **검증** — pliron이 모든 연산의 타입 일치, 블록 인자의 정확성, dominance를 확인합니다.
2. **최적화** — `mir-transforms`가 지원되는 주석 루프를 다시 쓰고 변경된 제어 흐름을 정리합니다.
3. **하위 변환** — `lower_mir_to_llvm`이 `mir.add`를 `llvm.fadd`로, `mir.load`를 `llvm.load`로, `mir.slice`를 포인터와 길이의 LLVM 구조체로 변환하는 등 작업을 수행합니다.
4. **필요한 경우 NVVM IR 준비** — `nvvm-transforms`가 연산을 선택된 libNVVM 타깃이 허용하는 LLVM 형식으로 변환합니다. 일반 PTX 빌드는 이 단계를 건너뜁니다.
5. **내보내기** — `vecadd`를 커널 진입점으로 표시하는 `!nvvm.annotations` 메타데이터를 포함해 LLVM dialect를 텍스트 `.ll` 파일로 출력합니다.
6. **컴파일** — 일반적으로 LLVM의 NVPTX 백엔드가 `.ll`을 `.ptx`로 컴파일합니다. NVVM 빌드는 libNVVM과 nvJitLink를 사용합니다. 결과는 호스트 바이너리 옆에 기록됩니다.

`dialect-mir`는 Rust 의미 체계를 충실히 보존합니다. 하위 변환 전에 cuda-oxide는 동작을 보존하는 표적 재작성을 수행할 수 있습니다. 첫 최적화는 [컴파일러 최적화](compiler-optimizations.md)에서 설명합니다. 이어서 [하위 변환 파이프라인](lowering-pipeline)이 LLVM에서 이해할 수 있는 형태로 바꿉니다.

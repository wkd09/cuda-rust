# Pliron — Pliron IR(MLIR 유사)

cuda-oxide는 Rust를 한 단계에서 곧장 PTX로 컴파일하지 않습니다. 각기 다른 추상화 수준을 나타내는 중간 표현을 거쳐 코드를 낮춥니다. 그 기반이 되는 **pliron**은 LLVM MLIR에서 영감을 받은 순수 Rust 확장형 컴파일러 IR 프레임워크이며, 이 위에 만든 IR을 **Pliron IR**이라 부릅니다. 이 장은 MLIR 개념, pliron을 선택한 이유, 핵심 자료 구조를 소개합니다. 컴파일러 경험은 필요하지 않습니다.

## MLIR이란? 왜 필요한가?

LLVM IR은 약 70개의 고정 연산(`add`, `load`, `br`, `getelementptr` 등)을 제공합니다. 도메인 개념이 이 연산에 맞지 않으면 높은 수준의 개념을 낮은 명령으로 평탄화해야 합니다.

**MLIR(Multi-Level Intermediate Representation)**은 고정 명령 집합 하나 대신 도메인별 명령 집합인 **dialect**를 여러 개 정의할 수 있는 프레임워크입니다. 각 dialect는 필요한 개념을 나타내는 연산, 타입, 속성을 모읍니다.

1. 도메인에 맞는 연산으로 dialect를 정의합니다.
2. dialect 간 변환(lowering) 패스를 작성합니다.
3. 백엔드가 받을 수 있는 표현에 도달할 때까지 패스를 연결합니다.

예를 들어 Triton은 Python GPU 코드를 텐서 수준 IR인 TTIR로 낮춰 “스칼라를 텐서 전체에 broadcast” 같은 연산을 일급 개념으로 둡니다. 이 수준에서 `splat + mul`을 벡터-스칼라 곱 하나로 바꾸는 도메인 최적화를 적용할 수 있습니다. LLVM IR로 평탄화한 뒤에는 원래 의미를 되살리기 어렵습니다.

cuda-oxide는 서로 다른 세 추상화 수준을 표현합니다.

| 계층 | 모델링 내용 | 연산 예 |
|:--|:--|:--|
| `dialect-mir` | 튜플, 열거형, 슬라이스, 검사 산술 등 Rust 의미 | `mir.extract_field`, `mir.get_discriminant` |
| LLVM dialect | 정수 산술, 메모리, 제어 흐름 등 기계에 가까운 연산 | `llvm.add`, `llvm.load`, `llvm.br` |
| `dialect-nvvm` | 스레드 인덱싱, 워프 셔플, TMA, WGMMA 등 GPU 인트린식 | `nvvm.read_ptx_sreg_tid_x`, `nvvm.shfl_sync` |

확장 가능한 IR이 없다면 Rust enum을 LLVM IR에 억지로 넣어 의미를 잃거나 계층마다 별도의 IR 프레임워크를 만들어야 합니다. MLIR식 dialect는 단일 시스템 안에서 세 표현을 정의하고 타입이 지정된 패스로 서로 변환합니다.

## pliron 소개

cuda-oxide에 필요한 것은 높은 수준 의미를 IR에 보존하고 GPU 백엔드가 받을 때까지 단계적으로 낮추는 모델입니다. upstream MLIR은 LLVM의 C++ 생태계에 속하지만 cuda-oxide는 Rust 크레이트, Rust 타입, Cargo 워크플로를 따릅니다.

[pliron](https://github.com/vaivaswatha/pliron)은 MLIR 개념을 따르는 순수 Rust 확장형 컴파일러 IR 프레임워크입니다. 연산, 영역, 기본 블록, 타입, 속성, dialect, 패스를 제공하면서 Rust 컴파일러 스택에 자연스럽게 통합됩니다.

| 항목 | pliron | upstream MLIR |
|:--|:--|:--|
| 언어 | 순수 Rust | Python 바인딩이 있는 C++ |
| 빌드 | `cargo build` | CMake와 LLVM 전체 빌드 |
| DSL | Rust 매크로만 필요 | TableGen, ODS |
| 디버깅 | Rust 도구(`dbg!`, rust-gdb) | C++ 템플릿용 gdb |
| 확장 | Rust 크레이트로 dialect 추가 | LLVM 헤더 기반 C++ 확장 |
| 의존 크기 | 배포 크레이트 하나 | 수 GB LLVM 산출물 |

MIR 가져오기부터 LLVM IR 내보내기까지 프로젝트의 Rust 빌드 및 디버그 환경을 유지할 수 있습니다.

## 핵심 자료 구조

### Context

`Context`는 연산, 블록, 영역, 타입, 속성, dialect 등록을 소유하는 중앙 자료 구조입니다. 컴파일 단위의 전체 IR을 담는 arena로 생각할 수 있습니다.

```rust
// 단순화한 형태. 실제 Context에는 더 많은 필드가 있습니다.
pub struct Context {
    operations: SlotMap<Ptr<Operation>, Operation>,
    basic_blocks: SlotMap<Ptr<BasicBlock>, BasicBlock>,
    regions: SlotMap<Ptr<Region>, Region>,
    dialects: HashMap<DialectName, Dialect>,
    type_store: TypeStore,     // 중복 제거된 타입
    // 속성은 중앙 저장하지 않습니다.
}
```

`Box` 힙 노드 대신 `slotmap`의 **세대 지정 arena**를 사용합니다. 삽입·삭제가 O(1)이고 인덱스가 안정적으로 유지되며, 슬롯 재사용 뒤 오래된 포인터는 세대 번호가 달라져 잘못된 메모리를 조용히 읽지 않고 런타임에 실패합니다. IR 데이터는 모두 `Context` 안에 있습니다. 함수 사이에 `&mut Operation`을 보관하는 대신 `Ptr<Operation>`을 들고 필요할 때 컨텍스트를 통해 역참조합니다.

### 연산

**연산(operation)**은 IR 그래프의 노드이며 피연산자(입력), 결과(출력), 컴파일 시점 속성, 선택적 중첩 영역을 가집니다. 각 연산은 `dialect_name.op_name` 형식의 이름을 사용합니다.

```rust
#[pliron_op(name = "mir.func", dialect = "mir")]
#[pliron_op(name = "nvvm.read_ptx_sreg_tid_x", dialect = "nvvm")]
#[pliron_op(name = "llvm.add", dialect = "llvm")]
```

`mir.func`는 MIR dialect의 함수 정의, `nvvm.read_ptx_sreg_tid_x`는 NVVM dialect의 스레드 ID 읽기, `llvm.add`는 LLVM dialect의 덧셈입니다. 각 연산 타입은 매크로가 생성하며 타입이 지정된 builder로 만들고 verifier로 피연산자·결과·속성을 검사합니다.

### 타입과 속성

타입은 `TypeStore`에서 유일화해 같은 타입 값이 공유됩니다. 타입 정의는 derive/속성 매크로로 선언합니다.

```rust
#[pliron_type(name = "mir.tuple", dialect = "mir")]
#[pliron_type(name = "mir.slice", dialect = "mir")]
#[pliron_type(name = "mir.enum", dialect = "mir")]
```

속성은 연산에 붙는 메타데이터입니다. 일부는 연산 필드에 직접 있고, 그 밖의 값은 타입/속성과 연결됩니다.

```rust
#[pliron_attr(name = "...", dialect = "...")]
```

### `Ptr<T>` — 안전한 arena 참조

`Ptr<T>`는 `Context`가 소유한 객체를 가리키는 세대 지정 핸들입니다. 일반 참조처럼 수명을 외부로 노출하지 않으므로 IR 노드끼리 연결하거나 변환 과정에서 핸들을 보관할 수 있습니다. 실제 접근에는 해당 `Context`가 필요하며, 삭제된 슬롯을 가리키는 오래된 핸들은 무효임을 검출합니다.

## 메모리 안전한 def-use 연결

연산 결과와 이를 사용하는 피연산자 관계를 **def-use 체인**이라 합니다. pliron은 결과 사용처와 피연산자 정의를 양방향으로 추적해 연산을 변경·삭제할 때 연결을 함께 갱신합니다. 변환 패스는 사용처를 찾기 위해 전체 모듈을 매번 훑을 필요가 없습니다. Rust 타입과 `Context` 접근 규칙 덕분에 IR 변경은 명시적인 API를 통해 수행됩니다.

```text
%sum = llvm.add %a, %b
%out = llvm.mul %sum, %factor
```

`%sum`은 `llvm.add`의 결과이자 `llvm.mul`의 입력입니다. `llvm.add`를 삭제하거나 결과를 교체할 때 사용처 연결도 일관되게 유지해야 합니다.

## 연산 인터페이스

연산 인터페이스는 여러 연산이 공유하는 동작 계약을 선언합니다. 예를 들어 terminator 인터페이스는 블록의 제어 흐름을 끝내는 연산을 표시하고, branch 인터페이스는 대상 블록과 전달 인자를 노출할 수 있습니다.

```rust
#[op_interface]
```

개별 연산은 해당 인터페이스를 구현합니다.

```rust
#[op_interface_impl]
```

이를 통해 최적화와 검증 코드는 구체적인 연산 타입마다 별도 분기를 쓰는 대신 공통 계약을 통해 블록 종결자, 분기 대상, 부수 효과 등을 다룰 수 있습니다.

## cuda-oxide에서 pliron 사용

주 lowering 경로는 로컬 `dialect-mir`, `dialect-nvvm`과 upstream `pliron-llvm`의 LLVM dialect 세 개를 사용합니다. 커널 이벤트 추적용 `dialect-iket`, 최종 구조화 PTX용 `dialect-ptx`도 별도로 있습니다. `Context::default`가 링크된 dialect, 연산, 타입, 속성 등록을 수행하므로 각 작성자가 직접 등록하지 않아도 됩니다.

### `dialect-mir` — Rust 의미

Rust MIR의 의미를 LLVM으로 직접 낮추며 잃을 정보를 보존합니다.

- `MirFuncOp`: Rust 함수 시그니처와 일치하는 블록 인자를 가진 디바이스 함수 진입점
- 산술/비교: Rust 오버플로 의미를 보존하는 검사/비검사 `mir.add`, `mir.sub`, `mir.eq`, `mir.lt` 등
- 집계 타입: `MirTupleType`, `MirStructType`, `MirEnumType`, `MirSliceType`, `MirArrayType` 및 `mir.extract_field`, `mir.get_discriminant`
- 메모리/제어 흐름: `mir.load`, `mir.store`, `mir.ref`, `mir.goto`, `mir.cond_br`, `mir.return`; `global`, `shared`, `local`, `tmem` 주소 공간 추적

### LLVM dialect — 기계에 가까운 IR

upstream `pliron-llvm`의 LLVM IR과 1:1로 대응하는 pliron 연산입니다. LLVM의 이진 연산 19개(`llvm.add`부터 `llvm.frem`), cast 연산 13개(`llvm.sext`, `llvm.trunc`, `llvm.bitcast` 등), `llvm.br`, `llvm.cond_br`, `llvm.switch`, `llvm.return`, `llvm.unreachable`을 포함합니다. 내보낼 때 블록 인자를 PHI 노드로 바꾸고 유효한 텍스트 `.ll`을 생성하며 `@llvm.used`, GPU 커널용 `!nvvm.annotations`를 기록합니다. cuda-oxide는 이 dialect와 내보내기 기능을 `llvm-export` 크레이트로 노출합니다.

### `dialect-nvvm` — GPU 인트린식

LLVM NVPTX 백엔드 인트린식을 타입이 지정된 pliron 연산으로 감쌉니다.

- 스레드 인덱싱: `nvvm.read_ptx_sreg_tid_x`, `nvvm.read_ptx_sreg_ctaid_x`, `nvvm.barrier0` 등은 `thread::index_1d()`의 기반입니다.
- 워프 원시 연산: 셔플(`nvvm.shfl_sync`), vote, reduction을 제공합니다.
- 가속기 연산: TMA bulk copy, Hopper WGMMA, Blackwell tcgen05 텐서 코어 명령을 제공합니다.

세 dialect 각각의 상세 내용은 [Pliron dialect](mlir-dialects.md) 장을 참고하세요.

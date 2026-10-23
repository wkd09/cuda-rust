# 컴파일러 최적화

cuda-oxide는 rustc와 LLVM의 최적화를 활용합니다. `mir-transforms`는 프로그램이 아직 `dialect-mir`로 표현되어 있을 때 cuda-oxide 고유 패스를 적용하는 크레이트이며, 현재 첫 패스는 루프 언롤링입니다.

## MIR 변환이란?

rustc가 Rust 코드를 MIR로 낮춘 뒤 cuda-oxide가 `dialect-mir`로 가져옵니다. 이 계층에는 서로 연관된 세 작업이 있습니다.

| 작업 | 설명 |
|:--|:--|
| **분석** | IR을 읽어 어떤 블록이 루프에 속하는지 등 사실을 기록합니다. |
| **변환** | 프로그램 계산 결과를 보존하면서 IR을 다시 씁니다. |
| **낮춤** | 한 IR dialect에서 더 낮은 수준 dialect로 변환합니다. |

루프 분석은 카운터, 증가량, 상한을 찾을 수 있습니다. 언롤링 변환은 이 정보로 루프 본문을 복제해 한 번에 여러 반복을 수행하도록 고칩니다. 이후 lowering이 LLVM으로 바꿉니다.

:::{note}
이 장의 **MIR 변환**은 cuda-oxide가 가져온 `dialect-mir`에 대한 변환을 뜻합니다. rustc 내부 MIR을 직접 수정하지 않습니다.
:::

## 변환이 실행되는 위치

```text
Rust 소스 → rustc MIR → mir-importer → dialect-mir → mem2reg
          → mir-transforms → mir-lower → LLVM dialect → LLVM IR → PTX
```

[ M I R importer](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/mir-importer.html)는 처음에 로컬 변수를 load/store로 표현합니다. `mem2reg`는 조건을 만족하는 로컬을 **SSA 값**으로 승격합니다. 각 값은 정의 지점 하나만 가지며 블록 인자가 블록 사이에 값을 전달합니다. 이 구조에서 카운터와 갱신을 분석하기 쉬워집니다. `dialect-mir`에는 Rust 타입과 연산이 남아 있어 표적 변환을 표현하기도 쉽습니다. 최적화 후에는 [낮춤 파이프라인](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/lowering-pipeline.html)이 결과를 변환합니다.

전체 변수 디버그 빌드는 `mem2reg`와 루프 언롤링을 건너뜁니다. 변수의 안정된 메모리 위치를 유지하면 cuda-gdb 디버깅이 쉬워집니다.

### 스택에 남은 로컬 진단

최적화된 verbose 빌드는 `llc`로 전달될 실제 LLVM IR도 확인합니다. 로컬의 스택 할당이 cuda-oxide `mem2reg`와 LLVM `-O2` scalar replacement 이후에도 남아 있으면 PTX 생성 전에 경고합니다.

```text
warning: local `scratch` (`[u32; 4]`, 16 bytes) in `kernel` could not be promoted to registers
  = note: indexed by a non-constant value; the allocation survives LLVM scalar replacement and remains stack-backed
  = help: use constant indexing or a small match/select when the index range is bounded
```

동적 인덱스는 선택된 원소를 알 수 없어서 작은 집계를 독립 SSA 값으로 바꾸기 어렵게 만들 수 있습니다. 주소를 다른 함수로 전달하거나 포인터로 저장·반환해 관찰 가능하게 만들어도 승격을 막습니다. 보수적인 패턴으로 원인을 특정하지 못해도 로컬을 알려주고 일반적인 설명을 제공합니다.

검사는 LLVM 최적화 성공 뒤 verbose 모드에서만 수행합니다.

```bash
cargo oxide build <example> --verbose
```

LLVM SROA가 나중에 제거할 할당을 cuda-oxide `mem2reg` 직후 진단하면 오탐이 생기므로 이 위치에서 검사합니다. 최적화가 꺼진 빌드와 전체 변수 디버그 모드에서는 의도적으로 저장소를 유지하므로 검사를 건너뜁니다.

## 첫 패스: 루프 언롤링

일반 루프는 매 반복마다 조건 검사와 되돌아가는 분기를 수행합니다. **언롤링**은 본문을 복사해 새 루프의 한 반복에서 원래 여러 반복의 작업을 수행하게 합니다.

```rust
let mut i = 0u32;
#[unroll]
while i < 4 {
    work(i);
    i += 1;
}
```

완전히 언롤링하면 개념상 다음과 같습니다.

```rust
work(0);
work(1);
work(2);
work(3);
```

조건과 루프 헤더 분기가 사라지고 각 본문 복사에서 `i`가 상수가 되어 추가 단순화가 가능합니다. 대신 생성 코드가 커지고 레지스터를 더 쓸 수 있으므로 언롤링은 성능 보장이 아닌 요청입니다.

## 전체 언롤링과 부분 언롤링

| 어노테이션 | 사용 시점 | 결과 |
|:--|:--|:--|
| `#[unroll]` | 반복 횟수를 컴파일 시점에 앎 | 반복마다 본문 하나를 만들고 루프를 제거 |
| `#[unroll(N)]` | 반복 수가 런타임에 결정될 수 있음 | 주 루프에 N개 복사본, 나머지 반복을 처리하는 remainder 루프 |

### 전체 언롤링

```rust
let mut i = 0u32;
#[unroll]
while i < 8 {
    sum += i;
    i += 1;
}
```

변환은 카운터가 0부터 7까지인 본문 여덟 개를 만들며 상수 전파가 리터럴에서 유도되는 식을 접을 수 있습니다.

### 부분 언롤링

```rust
let mut i = 0u32;
#[unroll(4)]
while i < n {
    process(i);
    i += 1;
}
```

`n`이 10이면 주 루프가 네 번씩 두 묶음을 처리하고 remainder 루프가 8, 9를 처리합니다. 나눠떨어지지 않아도 결과는 같습니다. 주 루프 조건 검사를 네 번 중 한 번으로 줄이고 더 긴 직선 코드 구간을 최적화에 제공합니다.

## 유용한 peephole: 파이프라인 단계 접기

부분 언롤링 뒤 주 루프 카운터는 런타임 값으로 남습니다. 흔한 소프트웨어 파이프라인에서 단계 선택을 상수화하도록 작은 **peephole 최적화**가 있습니다.

```rust
let mut k_idx = 0u32;
#[unroll(4)]
while k_idx < k_iters {
    let stage = k_idx & 3;
    match stage {
        0 => consume_stage_0(),
        1 => consume_stage_1(),
        2 => consume_stage_2(),
        3 => consume_stage_3(),
        _ => {}
    }
    k_idx += 1;
}
```

언롤된 주 루프 카운터는 4씩 증가하므로 그룹 사이 아래 두 비트는 바뀌지 않습니다. 따라서 복사별 `(k_idx + 0) & 3`, `(k_idx + 1) & 3`, `(k_idx + 2) & 3`, `(k_idx + 3) & 3`은 각각 0, 1, 2, 3으로 접힙니다. 상수 전파로 죽은 `match` 분기를 없애 네 단계별 직선 본문을 만듭니다. 나머지 반복용 루프는 유지합니다.

부호 없는 카운터의 `k_idx % 4`와 `(k_idx + 1) & 3` 같은 상수 오프셋도 인식합니다. 2의 거듭제곱 주기가 언롤링 간격을 나눌 때만 접습니다.

### 런타임 기준값이 접히지 않는 이유

식은 루프 카운터를 직접 기준으로 해야 합니다.

```rust
let stage = k_idx & 3;                    // 접을 수 있음

let global_k = tile_k_base + k_idx;
let stage = global_k & 3;                 // tile_k_base가 런타임 값이므로 접을 수 없음
```

`tile_k_base`의 하위 비트는 컴파일 시 알 수 없습니다. 이를 첫 형태로 바꾸려면 알고리즘이 `tile_k_base`가 4의 배수라고 별도로 보장해야 합니다. 패스는 이를 추측하지 않습니다.

:::{note}
전체 언롤링에서는 카운터가 이미 각 복사본에서 리터럴이므로 이 peephole 없이도 일반 상수 전파가 작동합니다.
:::

## 어노테이션이 패스에 도달하는 과정

1. `#[kernel]` 또는 `#[device]` 매크로가 `#[unroll]`, `#[unroll(N)]`을 소비해 루프 시작에 내부 표식을 넣습니다.
2. `mir-importer`가 호출을 인식해 런타임 코드를 만들지 않는 `mir.unroll_hint` 연산을 생성합니다.
3. `LoopInfo`가 루프를 찾고 induction 분석이 카운터, 시작값, 간격, 경계, 가능하면 반복 수를 구합니다.
4. 지원되는 여러 `continue` 경로를 헤더로 합치고 전체 언롤링의 루프 전달 값을 라우팅하는 등 제어 흐름을 정규화합니다.
5. 패스가 본문을 복제하고 전체 또는 부분 언롤링용 연결을 구성합니다.
6. 정리 단계에서 상수 접기, 분기 단순화, 죽거나 도달 불가능한 코드를 제거합니다.

표식이 없는 함수는 변경하지 않습니다. 안전성을 증명할 수 없으면 경고만 출력하고 언롤링하지 않으므로 루프 동작은 유지됩니다.

## 지원 루프와 안전한 대체

현재 패스는 보수적인 루프 형태를 인식합니다. 카운트 `while` 또는 정수형 카운터( `isize` 포함)의 범위 `for i in a..b`, `(a..b).rev()`에는 컴파일 시 상수 시작값, 헤더로 돌아오는 각 경로에서 동일한 증가량, 카운터와 경계를 직접 비교하는 부수 효과 없는 간단한 조건, 단일 일반 진입점이 필요합니다.

| 형태 | 현재 동작 |
|:--|:--|
| 명시적 카운트 `while` | 위 조건을 만족하면 지원 |
| 범위 `for` (`a..b`, `(a..b).rev()`) | 정수형 카운터 전체에서 지원 |
| 다른 반복자 (`a..=b`, `.step_by(k)`, `.iter()`) | 아직 인식하지 않으며 경고 |
| 상수 반복 수와 `#[unroll]` | 전체 언롤링 |
| 런타임 불변 상한과 `#[unroll(N)]` | 부분 언롤링 |
| 증가량이 같은 여러 `continue` 경로 | 경로를 합쳐 언롤링 |
| 조기 `break`/복수 종료를 포함한 전체 언롤링 | 지원 |
| 조기 `break`/다른 종료를 포함한 부분 언롤링 | 경고 후 건너뜀 |
| 중첩 루프 | 어노테이션이 있는 루프만 처리 |

부분 언롤링에는 `N >= 2`, 양수 카운터 간격, `<` 또는 `<=` 비교, 루프 안에서 변하지 않는 경계값도 필요합니다. 바깥 루프만 어노테이션했으면 안쪽 루프 복사본을 온전히 유지합니다. 둘 다 표시했으면 안쪽부터 처리합니다.

컴파일 시간과 메모리 사용을 제한하기 위해 어노테이션 하나가 만드는 본문 복사본은 최대 1,024개, 기본 블록은 8,192개, 연산은 65,536개입니다. 초과 요청은 경고 후 건너뜁니다. 변환하지 않은 루프의 경고에는 해당 `#[unroll]` 위치가 `file:line:column`으로 표시됩니다. 자세한 커널 작성 참고는 [루프 언롤링](https://nvidia.github.io/cuda-rust/cuda-oxide/gpu-programming/kernel-families.html#loop-unrolling)을 참조하세요.

## 언제 언롤링할까?

- 반복 수가 고정된 작은 루프
- 분기 오버헤드가 중요한 핫 카운트 루프
- 단계 선택이 상수화되는 소프트웨어 파이프라인
- 독립적인 본문 복사로 명령 수준 병렬성을 얻는 루프

본문이 크거나 레지스터 사용량이 이미 많으면 주의하세요. 코드 증가로 레지스터 압박, 점유율 저하, 명령 캐시 부담이 생길 수 있습니다. 알고리즘 구조에 맞는 작은 인수(예: 파이프라인 깊이)를 사용하고 성능을 측정하세요. `#[unroll]`을 제거한 결과와 비교해야 합니다.

```bash
cargo oxide build <example> --verbose
```

verbose 출력은 요청 인수와 패스가 찾은 루프 정보를 보여줍니다. 건너뛰면 경고 이유가 나타납니다. 루프 번호와 분석 필드는 내부 구현 세부 사항이라 바뀔 수 있습니다.

## 향후 변환을 위한 자리

```text
cuda-oxide/crates/mir-transforms/src/
├── analyses/
│   ├── loop_info.rs      # 각 루프의 블록은 무엇인가?
│   └── induction.rs      # 카운터, 간격, 경계, 반복 수는 무엇인가?
├── canonicalize.rs       # 제어 흐름을 단순한 형태로 정규화
└── unroll.rs             # 증명, 복제, 연결, 루프 정리
```

재사용 가능한 분석을 IR 변경과 분리하면 향후 최적화의 위치가 분명해집니다. 분석이 사실을 모으고 정규화가 동등한 형태를 맞춘 뒤 변환이 동작을 보존하며 IR을 바꿉니다. `mir-importer/src/pipeline.rs`가 패스 순서를 관리합니다. 각 패스는 변환 안전성을 증명하고 자원 사용을 제한하며 지원 범위를 벗어나면 동작을 바꾸지 않고 대체해야 합니다.

집중 테스트는 다음 명령으로 실행합니다.

```bash
cargo test -p mir-transforms --all-targets
```

다음 [낮춤 파이프라인](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/lowering-pipeline.html) 장에서는 최적화된 `dialect-mir`가 LLVM dialect와 PTX가 되는 과정을 설명합니다.

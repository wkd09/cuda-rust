# Pliron dialect들

cuda-oxide는 한 번의 변환으로 Rust를 PTX로 낮추지 않습니다. 각기 다른 추상화 수준을 모델링하는 세 가지 pliron dialect를 거칩니다. 두 가지(`dialect-mir`, `dialect-nvvm`)는 이 저장소에서 정의하고, LLVM dialect는 상위 `pliron-llvm` 크레이트가 제공합니다. 이 장에서는 세 dialect의 타입과 연산, 이들이 컴파일 파이프라인을 구성하는 방식을 살펴봅니다.

이 경로 외에도 저장소에는 여기서 다루지 않는 pliron dialect 두 개가 있습니다. `dialect-iket`은 커널 내부 이벤트 추적을 컴파일러에 표현하고, `dialect-ptx`는 직접 구성하거나 파싱된 PTX 소스에서 투영할 수 있는 구조화된 최종 PTX dialect입니다. 자세한 내용은 각 크레이트의 README를 참조하세요.

[Pliron — MLIR 유사 IR](pliron.md) 장을 아직 읽지 않았다면 먼저 읽어 보세요. 그 장의 개념(연산, 타입, 속성, 영역, `Ptr<T>`, def-use 체인)은 이 페이지의 기반입니다.

---

## 세 dialect 한눈에 보기

| Dialect          | 목적                      | 수준                                                            |
| :--------------- | :--------------------------- | :--------------------------------------------------------------- |
| **dialect-mir**  | Rust MIR 의미 체계 모델링    | 가장 높음 — Rust 타입, 튜플, 열거형, 슬라이스, 검사 산술 |
| **LLVM dialect** | LLVM IR 모델링               | 중간 — 평탄 타입, GEP, PHI 지원 제어 흐름                |
| **dialect-nvvm** | NVIDIA GPU 내장 기능 모델링 | 직교 — 스레드 인덱싱, 워프, TMA, WGMMA, tcgen05        |

LLVM dialect는 cuda-oxide 크레이트가 아닙니다. 연산, 타입, 속성, 연산 인터페이스 모델은 상위 `pliron-llvm` 크레이트에 있으며 cuda-oxide는 이를 의존성으로 사용합니다.

`dialect-nvvm`은 스택의 계층이 아니라 “직교”하는 dialect입니다. 연산이 LLVM dialect 아래가 아니라 LLVM 연산과 *나란히* 나타나기 때문입니다. 워프 셔플과 정수 덧셈이 같은 함수 본문에 함께 존재할 수 있습니다.

데이터는 파이프라인을 다음과 같이 통과합니다.

```text
dialect-mir ──(mem2reg)──▶ dialect-mir (SSA) ──(annotated unroll)──▶ dialect-mir
  ──(DialectConversion)──▶ LLVM dialect + dialect-nvvm ops ──(export)──▶ LLVM IR ──(llc)──▶ PTX
```

각 화살표는 pliron 또는 LLVM IR에 정의된 변환을 나타냅니다. 마지막 단계에서는 LLVM의 NVPTX 백엔드가 PTX를 생성합니다.

---

## dialect-mir — Rust 계층

`dialect-mir`는 Rust 타입 시스템과 제어 흐름 의미를 pliron 연산으로 보존합니다. 이는 의도된 설계입니다. 튜플, 열거형, 검사 산술, 주소 공간 같은 Rust 개념을 LLVM 타입 시스템으로 평탄화하기 *전*에 다루기 위함입니다.

### 타입

이 dialect는 Rust 타입을 반영하는 사용자 정의 타입 아홉 개를 정의합니다.

| 타입                 | 예시                                                   | 설명                                                   |
| :------------------- | :-------------------------------------------------------- | :------------------------------------------------------------ |
| `mir.tuple`          | `mir.tuple<i32, f32, i64>`                                | 서로 다른 타입을 담는 튜플                                          |
| `mir.ptr`            | `mir.ptr<f32, mutable, addrspace: 1>`                     | GPU 주소 공간을 지정하는 포인터                               |
| `mir.array`          | `mir.array<f32, 256>`                                     | 고정 크기 배열                                             |
| `mir.struct`         | `mir.struct<"Point", [f32, f32]>`                         | 레이아웃 정보를 담은 이름 있는 구조체                                |
| `mir.union`          | `mir.union<"Repr", [a, b], [i32, f32], 4, 4>`             | 모든 필드가 같은 바이트를 보는 Rust 유니온        |
| `mir.slice`          | `mir.slice<f32, addrspace: 1>`                            | 팻 포인터(포인터 + 길이)                                   |
| `mir.disjoint_slice` | `mir.disjoint_slice<f32>`                                 | 타입 지정 인덱스 공간과 범위 검사를 포함한 슬라이스              |
| `mir.enum`           | `mir.enum<"Option_i32", [("None", []), ("Some", [i32])]>` | 판별값과 variant 페이로드가 있는 Rust 열거형             |
| `mir.fp16`           | `mir.fp16`                                                | IEEE 754 binary16, Rust `f16`                               |

`mir.ptr`와 `mir.slice`의 주소 공간은 GPU 메모리 계층에서 데이터가 위치한 곳을 나타냅니다.

| 주소 공간 | 의미                                  |
| :------------ | :--------------------------------------- |
| 0             | 일반(런타임에 해석)            |
| 1             | 전역(디바이스 DRAM)                     |
| 3             | 공유(블록별 SRAM)                  |
| 4             | 상수(읽기 전용 캐시)               |
| 5             | 로컬(스레드별 스택, 넘치면 DRAM으로 스필) |
| 6             | 텐서 메모리(Blackwell TMEM)           |

### 연산

`dialect-mir`는 `cuda-oxide/crates/dialect-mir/src/ops/` 아래 모듈별로 12개 범주, 총 62개 연산을 정의합니다.

| 범주     | 예시                                                                                                                                                                                            | 개수 |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----: |
| Function     | `mir.func`                                                                                                                                                                                          |     1 |
| 제어 흐름 | `mir.goto`, `mir.cond_br`, `mir.return`, `mir.assert`, `mir.unreachable`, `mir.unroll_hint`                                                                                                         |     6 |
| 상수    | `mir.constant`, `mir.float_constant`, `mir.undef`                                                                                                                                                   |     3 |
| 메모리       | `mir.alloca`, `mir.load`, `mir.store`, `mir.assign`, `mir.ref`, `mir.ptr_offset`, `mir.memcpy`, `mir.memmove`, `mir.shared_alloc`, `mir.global_alloc`, `mir.extern_shared`                          |    11 |
| 산술   | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, `mir.checked_add`, `mir.checked_sub`, `mir.checked_mul`, `mir.neg`, `mir.not`, `mir.shr`, `mir.shl`, `mir.bitand`, `mir.bitor`, `mir.bitxor` |    15 |
| 비교   | `mir.eq`, `mir.ne`, `mir.lt`, `mir.le`, `mir.gt`, `mir.ge`, `mir.cmp`                                                                                                                               |     7 |
| 집계    | `mir.extract_field`, `mir.insert_field`, `mir.construct_struct`, `mir.construct_tuple`, `mir.construct_array`, `mir.construct_slice`, `mir.construct_disjoint_slice`, `mir.extract_array_element`, `mir.field_addr`, `mir.array_element_addr` |    10 |
| 열거형         | `mir.get_discriminant`, `mir.set_discriminant`, `mir.construct_enum`, `mir.enum_payload`                                                                                                            |     4 |
| 캐스트         | `mir.cast`                                                                                                                                                                                          |     1 |
| 저장소      | `mir.storage_live`, `mir.storage_dead`                                                                                                                                                              |     2 |
| 호출         | `mir.call`                                                                                                                                                                                          |     1 |
| 디버그        | `mir.dbg_value`                                                                                                                                                                                     |     1 |

연산이 많아 보이지만 자연스러운 범주로 나뉩니다. Rust MIR을 알고 있거나 [rustc_public 장](rustc-public.md)을 읽었다면 각 연산이 MIR 개념에 바로 대응합니다.

### IR 형태 예시

다음은 실제 `dialect-mir` 연산의 간단한 예시입니다. 읽기 쉽도록 단순화했으며 실제 출력에는 메타데이터가 더 포함됩니다.

**검사 덧셈**(Rust: `a, b: i32`일 때 `let sum = a + b`):

```text
// mir.checked_add returns a tuple (result, overflow_flag)
%checked = mir.checked_add %a, %b : i32
%sum     = mir.extract_field %checked, 0 : mir.tuple<i32, i1>
%overflowed = mir.extract_field %checked, 1 : mir.tuple<i32, i1>
%ok = mir.not %overflowed : i1
mir.assert %ok
mir.goto bb1
```

`mir.assert`는 조건이 참이면 블록 안에서 계속 진행하고, 그렇지 않으면 트랩을 발생시킵니다. 블록 병합으로 검사가 제거되지 않도록 분기와 분리되어 있습니다. LLVM 하위 변환은 어서션 위치에서 블록을 나누고 뒤따르는 연산 또는 트랩 블록으로 명시적으로 분기합니다.

**구조체 생성 및 필드 접근**(Rust: `point.x`):

```text
%point = mir.construct_struct %x, %y : mir.struct<"Point", [f32, f32]>
%x_val = mir.extract_field %point, 0 : mir.struct<"Point", [f32, f32]>
```

**공유 메모리 할당**(GPU 전용 기능):

```text
%shmem = mir.shared_alloc : mir.ptr<f32, mutable, addrspace: 3>
mir.store %value, %shmem : f32
```

### 검증

모든 MIR 연산은 생성 시 타입 일관성을 검증합니다. 이를 통해 가져오기 오류가 하위 변환 파이프라인으로 전파되어 몇 단계 뒤 난해한 LLVM 오류로 나타나기 전에 조기에 발견할 수 있습니다.

검사하는 항목의 예시는 다음과 같습니다.

- `mir.add`는 두 피연산자의 타입이 같은지 검사합니다.
- `mir.cond_br`는 조건이 `i1`(불리언)인지 검사합니다.
- `mir.extract_field`는 필드 인덱스가 범위 안에 있고 결과 타입이 필드 타입과 일치하는지 검사합니다.
- `mir.store`는 값의 타입이 포인터의 대상 타입과 일치하는지 검사합니다.

`DisjointSlice`는 일치하는 `ThreadIndex`만 받으므로 타입 시스템이 디바이스 측 인덱스 공간을 검사합니다. 호스트 실행 형상은 `PreparedLaunch<K>` 또는 unsafe 원시 실행 의무를 통해 고유성 증명을 완성합니다. 별도의 disjoint 접근 컴파일러 패스는 없습니다. 안전성은 이 API들에서 비롯됩니다.

---

## LLVM dialect — LLVM 계층

LLVM dialect는 LLVM IR을 pliron 연산으로 모델링합니다. 텍스트 `.ll` 파일과 거의 1:1로 대응합니다. 모든 LLVM 명령에 pliron 연산이 있고 타입도 LLVM 타입 시스템에 직접 대응합니다. dialect 자체(연산, 타입, 속성, 연산 인터페이스)는 상위 `pliron-llvm` 크레이트에서 정의합니다. cuda-oxide는 이를 가져와 얇은 `llvm-export` 크레이트를 통해 다시 내보냅니다. 이 크레이트에는 텍스트 `.ll` 내보내기와 pliron-llvm이 제공하지 않는 GPU 전용 확장(이름 지정 주소 공간, fp16 비트 헬퍼)도 있습니다.

### 타입

| 타입          | 예시                                 | 설명                                          |
| :------------ | :-------------------------------------- | :--------------------------------------------------- |
| 정수      | `i1`, `i8`, `i16`, `i32`, `i64`, `i128` | Pliron 내장 타입으로 직접 사용                       |
| 부동소수점        | `half`, `float`, `double`               | Pliron 내장 타입(`FP16Type`, `FP32Type`, `FP64Type`) |
| `llvm.ptr`    | `ptr addrspace(1)`                      | 선택적 주소 공간이 있는 불투명 포인터          |
| `llvm.struct` | `{ i32, float }` 또는 `%MyStruct`         | 이름이 있거나 없을 수 있고 불투명할 수도 있음                    |
| `llvm.array`  | `[256 x float]`                         | 고정 크기 배열                                    |
| `llvm.vector` | `<4 x float>`                           | SIMD 벡터                                         |
| `llvm.func`   | `(i32, ptr) -> void`                    | 함수 시그니처                                  |
| `llvm.void`   | `void`                                  | 단위 타입                                        |

Rust 전용 타입이 사라진 것을 확인하세요. 코드가 LLVM dialect에 도달할 때 튜플은 구조체가 되고 열거형은 판별값으로 인덱싱되는 구조체가 되며 슬라이스는 포인터와 길이 쌍이 됩니다. 이런 평탄화는 [하위 변환 파이프라인](lowering-pipeline) 패스에서 수행합니다.

### 연산

고정된 `pliron-llvm` 0.20.0 릴리스는 연산 69개를 정의합니다.

| 범주     | 예시                                                                                                                                | 개수 |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------- | ----: |
| 산술   | `add`, `sub`, `mul`, `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`, ...                                                                |    19 |
| 캐스트         | `zext`, `sext`, `trunc`, `fpext`, `fptrunc`, `sitofp`, `uitofp`, `fptosi`, `fptoui`, `ptrtoint`, `inttoptr`, `addrspacecast`, `bitcast` |    13 |
| 제어 흐름 | `br`, `cond_br`, `switch`, `return`, `unreachable`, `indirectbr`                                                                        |     6 |
| 메모리       | `load`, `store`, `alloca`, `gep`                                                                                                        |     4 |
| 원자 연산       | `atomic_load`, `atomic_store`, `atomicrmw`, `cmpxchg`, `fence`                                                                          |     5 |
| 비교   | `icmp`, `fcmp`                                                                                                                          |     2 |
| 집계    | `extract_value`, `insert_value`, `extract_element`, `insert_element`, `shuffle_vector`                                                  |     5 |
| 호출         | `call`, `call_intrinsic`                                                                                                                |     2 |
| 인라인 어셈블리   | `inline_asm`                                                                                                                            |     1 |
| 상수    | `constant`, `zero`, `undef`, `poison`                                                                                                   |     4 |
| 심볼       | `func`, `global`, `addressof`                                                                                                           |     3 |
| 선택       | `select`                                                                                                                                |     1 |
| 기타        | `freeze`, `va_arg`, `blockaddress`, `blocktag`                                                                                          |     4 |

`llvm-export`는 위 연산 외에도 자체 연산 `llvm.dbg_value` 하나를 추가하고 앞서 언급한 주소 공간 및 fp16 헬퍼를 제공합니다.

이 연산 목록은 고정된 상위 릴리스를 기준으로 합니다. 표는 0.20.0 크레이트에 기록된 소스 리비전인 [`pliron-llvm`의 `src/ops.rs`](https://github.com/pliron-org/pliron/blob/435536c8611af34c48c8f73d78e77894a05d103d/pliron-llvm/src/ops.rs)에 있는 연산을 셉니다.

LLVM IR을 본 적이 있다면 익숙한 내용입니다. 연산 이름은 LLVM 대응 항목과 같으며 IR에서는 앞에 `llvm.` 접두사를 붙입니다.

### 내보내기 엔진

`llvm-export`의 핵심은 pliron IR 모듈을 유효한 텍스트 LLVM IR로 변환하는 내보내기 모듈(`cuda-oxide/crates/llvm-export/src/export/`)입니다. 이 부분은 cuda-oxide가 자체적으로 관리합니다. pliron-llvm은 `llvm-sys` 브리지를 통해서만 실제 `.ll`을 출력하는데 cuda-oxide는 이를 사용하지 않습니다. 내보내기는 연산을 단순히 출력하는 작업이 아니며 여러 비자명한 변환을 수행합니다.

**블록 인자는 PHI 노드가 됩니다.** Pliron IR(MLIR 유사)은 병합 지점을 블록 인자로 모델링해 기본 블록 사이에 함수형 호출 규약을 둡니다. LLVM IR은 대신 PHI 노드를 사용합니다. 내보내기 엔진은 분기 피연산자로 선행 블록 맵을 만들고 진입점이 아닌 각 블록의 시작에 `phi` 명령을 출력합니다.

**값 이름 지정.** 사전 패스는 모든 값에 순차 SSA 이름(`%v0`, `%v1`, ...)을 붙입니다. 상수는 특별히 처리합니다. `llvm.constant` 결과는 `%vN` 이름 대신 리터럴 값으로 매핑하므로 PHI가 출력 뒤쪽에 나오는 블록의 상수를 참조할 수 있습니다.

**NVVM 내장 함수 이름 변환.** Pliron 식별자는 밑줄을 쓰고 LLVM 내장 함수는 점을 씁니다. 내보내기 엔진은 `llvm_`로 시작하는 이름에서 밑줄을 점으로 바꿉니다. 예를 들어 `llvm_nvvm_read_ptx_sreg_tid_x`는 `llvm.nvvm.read.ptx.sreg.tid.x`가 됩니다. 조회 테이블이 아닌 기계적인 변환입니다.

**Convergent 속성 표시.** 배리어, 셔플, 투표 내장 함수는 LLVM이 제어 흐름 밖으로 끌어올리지 못하도록 `convergent`로 표시해야 합니다. 내보내기 엔진은 점 형식 이름의 접두사 패턴을 확인해 해당 호출에 `#0`을 붙이고 모듈 수준에서 `attributes #0 = { convergent }`를 출력합니다.

**커널 메타데이터.** 커널로 표시된 함수는 `ptx_kernel` 호출 규약과 `!nvvm.annotations` 메타데이터 항목을 받습니다.

단순 벡터 덧셈 커널의 내보낸 LLVM IR은 다음과 같습니다.

```llvm
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define ptx_kernel void @vecadd(ptr addrspace(1) %v0, i64 %v1,
                                ptr addrspace(1) %v2, i64 %v3,
                                ptr addrspace(1) %v4, i64 %v5) {
entry:
    %v6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
    %v7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
    %v8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
    %v9 = mul i32 %v8, %v7
    %v10 = add i32 %v9, %v6
    ; ... bounds check, load, add, store ...
    ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @vecadd, !"kernel", i32 1}
attributes #0 = { convergent }
```

슬라이스가 스칼라화된 것을 확인하세요. Rust의 `&[f32]`는 `ptr addrspace(1)`과 `i64` 길이가 됩니다. 이는 하위 변환 패스에서 이루어지므로 LLVM dialect가 볼 때는 이미 평탄한 인자입니다.

---

## dialect-nvvm — GPU 계층

`dialect-nvvm`은 NVIDIA GPU 내장 함수를 타입 지정 pliron 연산으로 감쌉니다. 이 연산은 하위 변환 체인의 별도 “계층”을 이루지 않습니다. `dialect-mir`에서 LLVM dialect로 변환하는 패스 중 삽입되며 같은 함수 본문에서 LLVM dialect 연산과 공존합니다. 내보낼 때 `@llvm.nvvm.*` 내장 함수에 대한 `call` 명령이 됩니다.

### 아키텍처 지원 범위

카탈로그 SHA-256 `3f56e2a1`(모든 `ops/generated/` 파일 헤더의 표시값)을 기준으로 이 dialect는 42개 모듈에 585개 연산을 보유하며 두 경로에서 만들어집니다. 새 연산을 어디에 추가할지, 추가할 수 있는지를 결정하므로 이 구분을 먼저 알아야 합니다. 헤더 표시값이 더 이상 `3f56e2a1`로 시작하지 않는다면 이 페이지의 개수는 현재 읽는 카탈로그보다 오래된 것입니다.

**수작업 작성**은 `cuda-oxide/crates/dialect-nvvm/src/ops/` 바로 아래에 있습니다. 내장 함수 카탈로그가 설명하지 않는 전용 검증이나 하위 변환이 필요한 연산입니다. 모듈 7개, 연산 26개가 있습니다.

| 모듈    | 설명                                                 | 연산 |
| :-------- | :---------------------------------------------------------- | --: |
| `asm`     | `inline_ptx`                                                |   1 |
| `atomic`  | 원자적 load/store/RMW/cmpxchg/fence                         |   5 |
| `cluster` | 클러스터 인덱스 및 클러스터 수 레지스터                   |   2 |
| `debug`   | `assertfail`, `vprintf`                                     |   2 |
| `grid`    | 협력형 `grid_sync`                                     |   1 |
| `memory`  | 바이트 오프셋을 사용하는 일반 주소에서 공유 주소 변환     |   1 |
| `wgmma`   | 워프 그룹 MMA 디스크립터. m64n64k16에서 bf16/f16, m64n128k16에서 bf16, m64n64k8에서 tf32 |  14 |

**자동 생성**은 `cuda-intrinsics-gen`이 `intrinsics/catalog.json`에서 만들어 `ops/generated/`에 저장합니다. 모든 파일은 `// @generated ... DO NOT EDIT.`로 시작하며 직접 수정해도 다음 생성기 실행 때 덮어씌워집니다. 대부분의 연산이 여기에 해당합니다. 카탈로그 항목 1,038개에서 모듈 35개, 연산 559개를 생성합니다. 여러 내장 함수가 하나의 구조 연산을 공유할 수 있기 때문입니다.

| 영역                        | 모듈                                                                       | 연산 |
| :-------------------------- | :---------------------------------------------------------------------------- | --: |
| Tensor Core Gen 5 + TMEM    | `tcgen05`                                                                     | 210 |
| Tensor Memory Accelerator   | `tma`                                                                         | 116 |
| 특수 레지스터           | `sreg`                                                                        |  44 |
| 패킹(SIMD 레지스터 내)   | `packed_alu`, `packed_conversion`, `packed_atomic`                            |  55 |
| 비동기 복사와 배리어     | `cp_async`, `mbarrier_extended`, `mbarrier_basic`, `sync`                     |  35 |
| 워프 수준                  | `warp_shuffle`, `redux`, `vote`, `warp_match`, `warp_barrier`, `active_mask`, `elect` |  39 |
| 행렬 조각 이동    | `ldmatrix`, `register_mma`, `stmatrix`, `wgmma_control`, `movmatrix`, `sparse_mma` |  22 |
| 실행 및 디버그 제어 | `execution_control`, `debug_control`                                          |  11 |
| 클러스터                     | `clc`, `cluster_barrier`, `cluster_memory`                                    |  10 |
| 스칼라 수학                 | `dotprod`, `scalar_arithmetic`, `scalar_conversion`, `scalar_math`, `extended_minmax`, `prmt` |   9 |
| 정수 min/max(DPX)       | `integer_minmax`                                                              |   8 |

아키텍처 요구 사항은 모듈이 아니라 내장 함수별로 관리합니다. 카탈로그는 각 항목의 PTX 버전과 최소 SM을 기록하며 `intrinsics/generated-reference.md`는 예상 PTX와 함께 이를 표시합니다. 이 파일은 연산과 함께 재생성되므로 여기서 수작업으로 관리하는 개수보다 해당 목록을 참조하세요.

대부분의 사용자는 특수 레지스터, 워프 셔플, 배리어만 접하게 됩니다. 나머지는 TMA, 행렬 가속기, Blackwell 텐서 메모리 등 고급 GPU 프로그래밍용이며 [고급 GPU 기능](../advanced/tensor-memory-accelerator.md) 장에서 설명합니다. 연산을 추가한다면 거의 모든 새 내장 함수가 거치는 카탈로그 경로를 안내하는 [새 내장 함수 추가](adding-new-intrinsics)를 먼저 읽으세요.

### Rust에서 PTX까지: 내장 함수의 여정

각 NVVM 연산은 세 단계의 이름으로 대응합니다.

| Pliron 연산               | LLVM 내장 함수                                        | PTX 명령                        |
| :----------------------------- | :---------------------------------------------------- | :------------------------------------- |
| `ReadPtxSregTidXOp`            | `llvm.nvvm.read.ptx.sreg.tid.x`                       | `mov.u32 %r1, %tid.x`                  |
| `Barrier0Op`                   | `llvm.nvvm.barrier0`                                  | `bar.sync 0`                           |
| `ShflSyncBflyI32Op`            | `llvm.nvvm.shfl.sync.bfly.i32`                        | `shfl.sync.bfly.b32`                   |
| `CpAsyncBulkTensorG2sTile2dOp` | `llvm.nvvm.cp.async.bulk.tensor.2d.tile.g2s.im2col.*` | `cp.async.bulk.tensor.2d.tile.g2s ...` |

첫 열은 `dialect-nvvm`의 Rust 구조체 이름입니다. 둘째 열은 밑줄을 점으로 바꾼 뒤 `llvm-export`가 출력하는 이름이며, 셋째 열은 `llc` 결과입니다. 이를 수작업으로 작성할 필요는 없습니다. `mir-importer`가 `thread::threadIdx_x()` 또는 `warp::shuffle_xor_sync()` 같은 `cuda-device` 내장 호출을 변환할 때 연산을 출력하고 이후 단계가 나머지 두 형식을 만듭니다.

### 검증 전략

NVVM 연산은 최소한의 구조 검증을 수행합니다. 각 연산은 피연산자와 결과 개수를 검사하고 일부는 결과 타입도 확인합니다(스레드 인덱싱 연산은 `i32` 결과를 요구하고 `tcgen05` 로드는 32 레지스터 및 4 레지스터 변형에서 결과 개수가 정확한지 확인합니다).

이는 의도된 설계입니다. NVVM 연산은 `mir-importer`가 기계적으로 생성하므로 사용자가 직접 작성하지 않습니다. 이후 단계에서 LLVM NVPTX 백엔드가 포괄적인 타입 검증을 제공합니다. 모든 NVVM 연산에 전체 타입 검사를 넣으면 실질적 이득 없이 dialect 코드 크기만 두 배가 됩니다.

```{note}
GPU 아키텍처 요구 사항(sm_70, sm_90, sm_100)은 문서화되어 있지만 pliron 단계에서 강제되지는 않습니다. 아키텍처 검증은 특정 `-mcpu=sm_XX` 플래그로 `llc`를 호출할 때 수행됩니다. Hopper 내장 함수를 Volta 대상으로 사용하면 `llc`가 오류를 분명히 알려줍니다.
```

---

## dialect 간 상호 작용

Rust 연산 하나가 세 추상화 수준을 거치는 과정은 다음과 같습니다.

```text
Rust source:   let sum = a + b;        // a, b: f32

dialect-mir:   %sum = mir.add %a, %b : f32
                ↓  (DialectConversion)
LLVM dialect:  %v5 = fadd float %v3, %v4
                ↓  (llvm-export)
LLVM IR:       %v5 = fadd float %v3, %v4
                ↓  (llc --mcpu=sm_80)
PTX:           add.f32 %f3, %f1, %f2;
```

`dialect-mir`에서 LLVM dialect로 변환하는 단계에서 중요한 작업이 이루어집니다. `f32`의 `mir.add`는 부동소수점 덧셈 `fadd`가 되고 `i32`의 `mir.add`는 정수 덧셈 `add`가 됩니다. `mir.checked_add` 같은 검사 연산은 `llvm.add`, 오버플로 플래그용 상수 `i1 false`, 구조체에 넣는 `insertvalue`로 확장됩니다. GPU 경로에서는 정수 연산이 래핑되므로 오버플로 검사를 생략합니다. 하위 변환 패스가 이러한 변환을 처리합니다.

GPU 전용 연산에서는 `dialect-nvvm`이 사용됩니다.

```text
Rust source:   let tid = thread::threadIdx_x();
                ↓  (mir-importer matches "cuda_device::thread::threadIdx_x")
dialect-nvvm:  %v2 = nvvm.read_ptx_sreg_tid_x : i32
                ↓  (mir-lower, then llvm-export)
LLVM IR:       %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
                ↓  (llc)
PTX:           mov.u32 %r1, %tid.x;
```

`mir-importer`의 변환기는 정규화된 전체 이름(FQDN)으로 `cuda_device` 내장 함수 호출을 인식합니다. `threadIdx_x`와 일치하는 분기는 `intrinsics/catalog.json`에서 디스패치 코드로 생성됩니다. 호출은 `mir.call`이 되지 않습니다. 변환기가 즉시 `dialect-nvvm` 연산을 출력하므로 해당 내장 함수는 직접 하드웨어 명령으로 이어집니다.

### 전체 모습

전체적으로 컴파일된 커널 본문에는 LLVM dialect 연산과 `dialect-nvvm` 연산이 함께 들어 있습니다.

```text
llvm.func @vecadd(...) {
  entry:
    %tid    = nvvm.read_ptx_sreg_tid_x     // NVVM: thread index
    %ntid   = nvvm.read_ptx_sreg_ntid_x    // NVVM: block size
    %ctaid  = nvvm.read_ptx_sreg_ctaid_x   // NVVM: block index
    %offset = llvm.mul %ctaid, %ntid        // LLVM: integer math
    %idx    = llvm.add %offset, %tid        // LLVM: integer math
    %cmp    = llvm.icmp slt %idx, %len      // LLVM: bounds check
    llvm.cond_br %cmp, bb1, bb2             // LLVM: branch
  bb1:
    %p_a    = llvm.gep %a, %idx             // LLVM: pointer arithmetic
    %val_a  = llvm.load %p_a                // LLVM: memory access
    %p_b    = llvm.gep %b, %idx
    %val_b  = llvm.load %p_b
    %sum    = llvm.fadd %val_a, %val_b      // LLVM: floating-point add
    %p_c    = llvm.gep %c, %idx
    llvm.store %sum, %p_c
    llvm.br bb2
  bb2:
    llvm.return void
}
```

맨 위의 `dialect-nvvm` 연산은 전역 스레드 인덱스를 계산합니다. 나머지는 로드, 저장, 산술, 분기 같은 표준 LLVM dialect 연산입니다. 내보내기 엔진은 모두 하나의 `.ll` 파일로 직렬화하고 `llc`가 PTX로 컴파일합니다.

---

하위 변환 패스가 이 dialect들을 연결하는 방식은 [하위 변환 파이프라인](lowering-pipeline)을 참조하세요.

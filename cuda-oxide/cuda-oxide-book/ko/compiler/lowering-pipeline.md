# 하위 변환 파이프라인

이전 장들은 IR를 구축하고 준비했습니다.[MIR 수입자](mir-importer.md)으로 번역된`dialect-mir`- [컴파일러 최적화](compiler-optimizations.md)어떻게 설명했지`mir-transforms`IR를 다시 쓰면,[플리론 방언](mlir-dialects.md)설명된 것`dialect-mir`, LLVM 방언, 그리고 `dialect-nvvm`.

이 장은 다음 브릿지를 다루고 있습니다. Rust-flavored IR를 LLVM가 컴파일할 수 있는 것으로 바꾸는 것입니다.

NVVM 빌드에서는 별도의 `nvvm-transforms`이 아래로 떨어지는 후 통과 실행. 그것은 유지 `mir-lower`전환에 집중합니다`dialect-mir`LLVM 방언으로, 텍스트 수출 전에 레거시 또는 현대 libNVVM에 해당 LLVM 모듈을 조정합니다.

만약 당신이  타입을 알고 있다면, LLVM에서 들어 본 적이 없는 의 숫자를 알게 될 것입니다.

---

## 낮출 것 은 무엇 을 의미 합니까?

`dialect-mir`은 톱셀, 엔룸, 슬라이스, 체크 아리듬틱, GPU 주소 공간에 대해 알고 있습니다. LLVM IR는 그 어떤 것도 모릅니다. 평형 정수와 플로트 타입이 있습니다.`getelementptr`, PHI 노드, 그리고 하나의 수준의 추상화 이상의 모든 것에 대한 일반적인 의심.

**하하하는 것은 모든 을 교체하는 과정입니다.`dialect-mir`LLVM 방언 연산의 동등한 순서로 동작하는 것, 하나씩,`dialect-mir`연산은 유지된다. 튜플은 익명 구조가 된다. 슬라이스는 포인터 길이의 쌍이 된다. 체크된 덧셈은 LLVM 오버플로우 내장 함수가 되고 추출이 뒤따른다. 모든  개념은 LLMV가 소화할 수 있는 무언가에 납작해진다.

이 모든 작업을 수행하는 패스는 `cuda-oxide/crates/mir-lower/`에 있으며, 플리론의 `DialectConversion` 프레임워크를 사용합니다. 파이프라인에서 가장 큰 단일 변환이며, 이 장의 나머지 부분은 어떻게 작동하는지에 관한 것입니다.

---

## 방언 변환 -- 하락 프레임

이 로저는 수동 워크 앤 리프레이스 패스 대신 플리론의 `DialectConversion` + `DialectConversionRewriter`를 사용합니다. 프레임워크는 IR 워킹, def-before-use 순서, 타입 변환 및 블록 논증 패치를 자동으로 처리합니다.

### 어떻게 작동 합니까?

각 `dialect-mir`그리고 `dialect-nvvm`op는 을 통해 자신을 하위 변환 방법을 선언`MirToLlvmConversion`op 인터페이스 (에서 정의)`conversion_interface.rs`이 인터페이스는 하나의 방법을 가지고 있습니다:`convert(ctx, rewriter, op, operands_info)`모든 작전의 실행은`convert/interface_impls.rs`, 이는 카테고리별로 구성된 변환기 기능에 전송합니다.

모듈의 각 `MirFuncOp`에 대해, `convert_func` (`lowering.rs`에서):

1. ** LLVM 함수를 생성합니다** 그 서명은
함수는 커널 입구점이다. 커널이 아닌 함수는 내부 CUDA ABI를 위한 모든 집적 arg을 스칼라 필드로 평평하게 한다. 커넬 입구점은 슬라이스를 평평히 한다 (`(ptr, len)`하지만 호스트 발사자가 전체 집합체를 하나의 패킷 슬롯으로 밀어주기 때문에 하나의 바이발 값으로 구조와 폐쇄를 통과합니다.{ref}`Argument Scalarization <lowering-argument-scalarization>`아래에서 참조하십시오.)
2. ** GPU 메타데이터를 퍼포게팅합니다** (`gpu_kernel`, `maxntid`,`cluster_dim_*`).
3. ** MIR 블록을 LLVM 함수로 이동하기 위해 `inline_region`**를 사용합니다.
원래 블록이 보존됩니다. 수동 블록 매핑이 필요하지 않습니다.
4. ** MIR 레벨 값을 재구성하는 입력 프롤로그를 만듭니다
LLVM 수준에 도달한 모든 매개 변수에서 어떤 형태이든 (내부 호출 및 슬라이스용 평평 스칼라의 `insertvalue`; 커널 경계에서 바이발 집계수에 대한 통과).
5. ** 모든 MIR 동작을 실행하고 호출하는 `DialectConversion`**를 실행합니다
이 LLVM 운영에 대입하기 위해 이 `MirToLlvmConversion::rewrite` 구현.

### 변환 모듈

변환 함수는 카테고리별로 모듈로 구성됩니다:

|카테고리      | 모듈                        | 이 책 의 내용|
| :------------ | :---------------------------- | :--------------------------------------------------------------------------------|
|수학적    | `convert/ops/arithmetic.rs`   | `add`→`add`, `sub`→`sub`, `checked_add`→`add`+`extractvalue`|
|메모리력        | `convert/ops/memory/*.rs`     | `mir.load`→`load`, `mir.store`→`store`, `shared_alloc`→글로벌 + `addrspacecast`|
|제어 흐름  | `convert/ops/control_flow.rs` | `mir.goto`→`br`, `mir.cond_br`→`cond_br`, `mir.return`→`return`|
|합산     | `convert/ops/aggregate/*.rs`  | 구조/투플 필드 액세스 → GEP 또는 `extractvalue`/`insertvalue`|
|캐스팅          | `convert/ops/cast.rs`         | `IntToInt`→`zext`/`sext`/`trunc`, `FloatToFloat`→`fpext`/`fptrunc`, 등|
|전화해          | `convert/ops/call.rs`         | `mir.call`→`call`, 논쟁을 평평화하고 `::``__`이름 변환|
|GPU 내성 | `convert/intrinsics/*.rs`     | NVVM ops → LLVM 내장 함수 호출 또는 직렬 PTX|
|변수     | `convert/ops/constants.rs`    | `mir.constant`→`llvm.constant`|

프레임워크는 자동으로 이러한 것을 전송합니다. 각 OP의 `MirToLlvmConversion` 임플로트는 올바른 변환기 기능을 호출합니다. 복잡성은 각 변환기 안에 살고 있으며, 여기서 Rust 의미론은 LLVM 현실을 충족합니다.

---

## 타입 변환

연산이 변환되기 전에, 그 타입이 변환되어야 합니다. LLVM의 타입 시스템은 Rust보다 의도적으로 단순합니다. 정수, 튜플, 엔룸, 지방 포인터에 대한 서명도 없습니다. 모든 것이 평평해야합니다.

|MIR 타입                       | LLVM 타입                    | 참고문서|
| :----------------------------- | :--------------------------- | :----------------------------------------------------------------------------- |
|`IntegerType(32, Unsigned)`    | `IntegerType(32, Signless)`  | LLVM 정수들은 표지를 가지고 있지 않습니다. 표기된 것은 * 연산에 있습니다.|
|`MirTupleType<i32, f32>`       | `{ i32, float }`             | 튜플은 익명 구조가 됩니다|
|`MirSliceType<f32>`            | `{ ptr, i64 }`               | 지방 포인터 분해 -- 포인터 + 길이|
|`MirStructType`                | `{ fields..., [N x i8] }`    | 의 레이아웃에 맞게 명시적인 포딩 배열|
|`MirPtrType<f32, addrspace:3>` | `ptr addrspace(3)`           | 주소 공간 보존된 불투명한 포인터|
|`MirArrayType<f32, 256>`       | `[256 x float]`              | 직접 매핑 -- 배열은 LLVM에도 충분히 간단합니다|
|`MirEnumType`                  | `{ discriminant, [M x i8] }` | 가장 큰 변종에 맞게 크기를 결정하는 + 유해물|

정수 서명 사례는 강조를 받을 가치가 있습니다.`i32`그리고 `u32`LLVM에서는 둘 다 입니다.`i32`표지 정보가 작업으로 이동합니다:`icmp slt`, 서명되지 않은 하나는 `icmp ult`타입 변환기는 표기성을 떨어뜨리고, 연산 변환기는 비교와 분할 명령어를 발산할 때 다시 잡습니다.

(하하-항용-스칼라리제이션) =

### 논증 스칼라리제이션

커널 진입점는 특별한 치료가 필요합니다. CUDA 드라이버는  패트 포인터를 이해하지 않습니다.`&[f32]`ABI의 양쪽에 별도의 포인터와 길이로 도착해야 합니다. 다른 한편으로, 값에 따라 구조와 닫는 것은 단일 호스트 패킷 슬롯과 일치합니다. 따라서 커널 항목은 하나의 바이발로 `.param`-- 그렇지 않으면 장치가 N의 평평한 논리를 기대하고 호스트는 하나만 누르며, 후속 슬라이스를 매번 부합하지 않습니다.

따라서 하락 패스는 커널 엔트리 규칙과 내부 호출 규칙을 구별합니다:

|MIR 커널 파라      | LLVM 커널 서명|
|:----------------------|:-------------------------------------------|
|`&[f32]`              | `ptr addrspace(1) %ptr, i64 %len`|
|`T` (스칼라)          | 지나가면|
|`struct { a, b }`     | 한 개의 부가 값 `{a, b}` 값|
|포획 (N 포획)  | 1개의 부근 폐쇄 구조 값|
|0 크기의 합물  | 떨어졌어요 (LLVM arg, 호스트 패킷 슬롯이 없네요)|

슬라이스 케이스는 여전히 엔트리 블록에서 평평한 패턴에서 고전적인 재구성을 사용합니다:

```text
MIR:  fn kernel(slice: &[f32])
      → entry arg: %slice : MirSliceType

LLVM: fn kernel(ptr addrspace(1) %ptr, i64 %len)
      → entry block reconstructs:
          %slice  = insertvalue {ptr, i64} undef, %ptr, 0
          %slice2 = insertvalue {ptr, i64} %slice, %len, 1
```

구조/폐기 경우 재구성을 건너뛰고, 바이발 값은 이미 올바른 형태이고, 나머지 함수는 특별한 일이 일어나지 않은 것처럼 그것을 봅니다. 내부 장치-기기 호출은 항상 같은 방식으로 집합을 평평화합니다. 따라서 이 규칙의 비용은 커널 경계에서만 발생합니다.

---

## 흥미 로운 회심

대부분의 변환은 간단합니다: 정수에서 `mir.add`는 `llvm.add`, `mir.load`는 #`llvm.load`, #`mir.goto`는 @`llvm.br`로 변합니다. 흥미로운 경우는 단일 MIR 연산이 여러 LLVM 연산으로 확장되는 경우 또는 GPU 특수한 문제가 번역을 완전히 변경하는 경우입니다.

### 수학적 검증

디버그 빌드에서, Rust는 오버플로우를 위해 모든 정수 수학적 연산을 검사합니다. MIR는 이 같은 연산을 `mir.checked_add`와 같이 모델링하여 `(result, overflow_flag)` 튜플을 반환합니다. LLVM에는 그러한 개념이 없지만 오버블로우 내장 함수가 있습니다:

```text
MIR:  %result = mir.checked_add %a, %b : i32  → mir.tuple<i32, bool>

LLVM: %sum = add i32 %a, %b
      %overflow = extractvalue {i32, i1} @llvm.sadd.with.overflow.i32(%a, %b), 1
```

오버플로우 플래그는 MIR 수입자가 이미 도달할 수없는 블록을 목표로 한 조건부 브랜치로 낮아졌다는 주장으로 먹이를줍니다. GPU에서 이것은 효과적으로 의미합니다. 정수 오버풀로를 유발하는 데 성공하면 커널이 함락됩니다. 가장 우아한 오류 처리 방식이 아니지만 CUDA 도구 체인은 오늘날 스택 풀기를 지원하지 않습니다.

### 공유 메모리

CUDA의 공유 메모리는 블록 스코프 SRAM입니다. 빠르고 작고 정적으로 선언됩니다. `dialect-mir`에서는 `mir.shared_alloc` 동작입니다. LLVM IR에서는 공유 메모리가 주소 공간 3:의 모듈 수준의 글로벌 변수여야합니다.

```text
MIR:  %shmem = mir.shared_alloc : mir.array<f32, 256>

LLVM: @shmem_0 = addrspace(3) global [256 x float] zeroinitializer
      %ptr = addrspacecast [256 x float] addrspace(3)* @shmem_0 to ptr
```

이 함수는 `addrspacecast`가 주소 공간에 대해 걱정하지 않고 나머지 함수가 사용할 수 있는 일반 포인터를 생성합니다. LLVM의 NVPTX 백엔드는 나머지를 처리합니다. 그것은 `addrspace(3)`가 공유 메모리를 의미한다는 것을 알고 적절한 `st.shared` / `ld.shared` 명령어를 생성합니다..

### 

엠럼은 대수적으로 풍부하다. LLVM에는 태그 연합의 개념이 없다. 낮출 패스는 두 개의 필드를 가진 구조로 엔럼을 표현함으로써 격차를 줄여줍니다.

```text
MIR:  %opt = mir.construct_enum "Some", (%val) : mir.enum<"Option_i32">

LLVM: %tmp = insertvalue { i8, [4 x i8] } zeroinitializer, i8 1, 0
      %result = insertvalue { i8, [4 x i8] } %tmp, <val into payload area>
```

분별하는 것은`i8 1`왜냐하면`Some`의 변수 1`Option`- 화물량은`[4 x i8]`-- 4바이트, 을 저장할 수 있을 만큼`i32`. 변수 접근은 역으로 작동합니다: 차별자를 읽고, 그것에 갈라지십시오, 그리고 `extractvalue`유료 화면과 비트캐스트가 예상되는 타입으로

그것은 우아하지 않지만 C 컴파일러가 수십 년 동안 태그 조합을 처리 한 방식입니다. LLVM의 최적화는 불필요한 삽입값 / 추출값 사슬을 청소하는 데 매우 좋습니다.

태그와 유료 부하 바이트가 저장소에 미흡한 엔룸, 예를 들어 `repr(align(N))` 엔룸은 또한 정렬 앵커로 슬롯 0에서 제로 길이의 정수 배열을 얻습니다. 앵크는 바이트를 추가하지 않으며 16 바이트로 제한됩니다. 러스트 정렬 자체는 엔룸의 모든 할당, 부하 및 저장소에 스탬프됩니다.

```{figure} images/enum-alignment-anchor.svg
:align: center
:width: 100%

enum 정렬 앵커. 슬롯 0에 있는 제로 길이의 정수 배열은 바이트를 추가하지 않고 저장 형식의 자연 정렬을 높인다. 그 너비는 제한 없이 `align * 8`로 사용되었는데, LLVM은 정렬 2^21에서 거부하고 2^29에서 `i0`으로 엮는다. 현재는 `[0 x i128]`로 제한되어 있으며, #610 이후 유니온 스토리지가 되었으며, 러스트 정렬은 모든 메모리 작업에 탑재됩니다.
```

---

## GPU 내성 변환

`dialect-nvvm`스레드 인덱싱, 워프 셔플, 바리어, TMA 대용량 복사 -는 일반 LLVM 방언 작업으로 낮춰지지 않습니다. LLVM 내장 함수 호출 또는 직선 PTX 어셈블리로 낮춰집니다.

### 전략 1: LLVM 내장 함수 통화

LLVM이 이미 목표 특정 내장 함수 인 것을 제공하는 작업에 대해 변환은 해당 내장 함수인 것으로 `call`를 방출합니다:

```text
nvvm.read_ptx_sreg_tid_x
  → call i32 @llvm_nvvm_read_ptx_sreg_tid_x()

nvvm.shfl_sync_bfly_i32
  → call i32 @llvm_nvvm_shfl_sync_bfly_i32(i32 -1, i32 %val, i32 %mask, i32 31)
```

워프 셔플을 보세요: 사용자로 향하는 `cuda_device`API는 두 개의 인수를 (value와 lane mask) 취하지만 LLVM 내장 함수는 네 개 (membermask, value, delta, clamp) 를 취합니다. 낮출 패스는 부족한 인수를 채우죠 -- `membermask = -1`(모든 차선) 그리고 `clamp = 31`(전형폭) -- 그래서 사용자는 그것에 대해 생각할 필요가 없습니다.

### 전략 2: 직선 PTX 조립

최신 GPU 명령어는 종종 LLVM 내장 함수성을 가지고 있지 않습니다. 이 경우, 하위 변환 패스는 LLVM의 `asm` 문법을 사용하여 직선 PTX 어셈블리를 방출합니다:

```text
nvvm.wgmma_fence_sync
  → call void asm sideeffect convergent "wgmma.fence.sync.aligned;", ""()

nvvm.mbarrier_arrive
  → call i64 asm sideeffect convergent "mbarrier.arrive.shared.b64 $0, [$1];", "=l,r"(ptr %bar)
```

`convergent`이 속성은 여기서 매우 중요합니다. 이 속도는 LLVM에 "이 명령어를 제어 흐름에 이동, 복제 또는 추측하지 마십시오".라고 말합니다. 이 기능이 없으면 LLVM은 조건부 지부에서 장벽을 올릴 수 있습니다. 또는 워크 레벨 명령어를 동기화 지점을 넘어 가라앉히게 할 수 있습니다, 결과적으로 GPU가 쓰레기를 매달거나 계산합니다. 이 둘 중 어느 것도 유용한 오류 메시지를 생성하지 않습니다.

---

## PHI 노드에 대한 블록 논증

플리론 IR (MLIR와 비슷) 는 기본 블록 사이의 값 흐름에 ** 블록 논리를 ** 사용합니다. LLVM은 ** PHI 노드를 ** 사용합니다 . 그들은 같은 개념을 표현합니다. "이 값은 다른 전신에서 온다". 그러나 문법은 수출 단계가 단지 예쁜 인쇄가 아니라 실제 변환이 필요하기에 충분합니다.

플리론 스타일 (블록 논증):

```text
^loop_header(%sum: f32, %i: i64):
    ...
    br ^loop_header(%new_sum, %new_i)
```

LLVM IR 스타일 (PHI 노드):

```text
loop_header:
    %sum = phi float [ 0.0, %preheader ], [ %new_sum, %body ]
    %i = phi i64 [ 0, %preheader ], [ %new_i, %body ]
```

수출자는 이 변환을 두 가지 접근법으로 처리합니다:

1. **예제 승격: 모든 값을 지정합니다. ** 어떤 코드도 발급하기 전에 수출자는 발주합니다.
모든 블록을 지정하고 순차적인 SSA 이름을 부여합니다 (`%v0`, `%v1`, ...) 에 대한 모든 값. 이것은 PHI 노드가 목록에서 * 나중에 * 나타나는 블록에서 값을 참조할 수 있기 때문에 중요합니다. 루프 뒷 가장자리는 텍스트에서 앞으로 가리키지만 제어 흐름에서 뒤로 가리키고 있습니다. 사전 명칭이 없으면 이러한 참조는 정의되지 않습니다.

2. **전선 지도를 구축합니다.
함수에서 모든 브랜치 명령어를 검사하여 `(predecessor_block, values_passed)` 페어를

3. **PHI 노드를 발산합니다.** 각 비입국 블록의 입시에 수출자는 발산 합니다
블록 논증당 PHI 노드 하나, 전신 지도의 값과 전신 레이블로 채워집니다.

선행은 미묘한 부분이다. 루프 헤더의 PHI는 루프 본부에서 `%new_sum`를 참조하지만, 루프 몸은 텍스트 출력에서 헤더 * 뒤에 * 나타난다. 만약 우리가 발사 중에 비행에 이름을 부여한다면, `%new_sum`는 아직 이름을 갖지 못할 것이다. 선행이 모든 것을 앞당겨서 이름을 붙임으로써 이 문제를 제거한다.

---

## 디버그 위치

rustc는 이미 대부분의 MIR 동작이 소스 프로그램에서 어디에서 왔는지 알고 있습니다. cuda-oxide는 커널을 하위 변환 동안 그 정보를 보존합니다:

```text
rustc Span
  ↓
pliron Location
  ↓
LLVM !dbg metadata
  ↓
PTX .loc / DWARF
  ↓
cuda-gdb source line
```

일반적인 경우는 함수와 같은 파일의 소스 라인입니다:

```llvm
define ptx_kernel void @vecadd(...) !dbg !func {
  %idx = call i64 @cuda_device____internal__index_1d(...), !dbg !loc
}

!loc = !DILocation(line: 39, column: 13, scope: !func)
```

"이 명령어는 이 함수의 13번 열의 39줄에 속한다". Cuda-gdb가 오른쪽  줄에 멈출 수 있을 뿐입니다.

### 크로스 파일 스펜

일부 소스 스판은 도움말 파일로 가리키고 있습니다. 예를 들어, 의 커널`src/main.rs`에 들어갈 수 있습니다`cuda-device/src/thread.rs`전화할 때`thread::index_1d()`.

수출자는 첨부하지 않아야 합니다`thread.rs:292`바로 커널의`src/main.rs`그 범위는 디버거에 "main.rs의 292호선"을 알려줍니다.

대신, cuda-oxide는 파일 특정 디버그 범위를 만듭니다:

```text
src/main.rs function scope
  |
  +-- cuda-device/src/thread.rs file scope
        |
        +-- line 292, column 19
```

LLVM 메타데이터에서는 이렇게 보입니다:

```llvm
!thread_scope = !DILexicalBlockFile(scope: !kernel_scope, file: !thread_file, discriminator: 0)
!loc = !DILocation(line: 292, column: 19, scope: !thread_scope)
```

플리론이 명시적인 호출 지점 위치를 제공하면 수출자는 또한 도움말 코드에서 어디서 연락했는지 기록합니다:

```llvm
!call = !DILocation(line: 39, column: 13, scope: !kernel_scope)
!loc = !DILocation(line: 292, column: 19, scope: !thread_scope, inlinedAt: !call)
```

"우리는 `thread.rs:292`, `main.rs:39`에서 도달한 [[$S0$]]를 실행하고 있습니다".

### 변수

풀 디버그 모드는 라인 위치 위에 첫 번째 변수 레코드를 추가합니다:

```llvm
%tid_slot = alloca i32
call void @llvm.dbg.declare(metadata ptr %tid_slot, metadata !tid, metadata !DIExpression())

!tid = !DILocalVariable(name: "tid", scope: !func, file: !file, line: 31, type: !u32)
```

모델은 모든 디버그 빌드가 사용하는 모델입니다:

```text
dbg.declare -> "this source variable lives at this address"
```

주소가 변수의 전체 범위에서 안정적이기 때문에 cuda-gdb는 그 범위 내의 모든 중단 지점에서 읽을 수 있습니다. 그 주소를 실제로 유지하려면 풀 디버그가 있습니다.`-G`- 스타일 빌드: 그것은 ** skips** 플리론 `mem2reg`(그렇게 스택 슬롯이 살아남을 수 있습니다.) ** skips** LLVM `opt -O2`, 그리고 실행 `llc``-O0`. SSA 값으로 로컬을 촉진하면 검사 가능한 범위가 레지스터의 활력으로 줄어들 것입니다. 최적화된 빌드가 어떻게 지역적 범위를 표시하는지는 정확히`<optimized out>`.

집계 로컬 (스트럭트, 툴플, 배열) 는 구성원들이 rustc의 실제 레이아웃 오프셋을 가지고 있는 `DICompositeType`를 얻으며, 따라서 `info locals`는 그들의 필드를 인쇄합니다:

```text
out = DisjointSlice {ptr: 0x..., len: 1}
```

플리론`mem2reg`여전히 프로모션에 대한 인식 구조 경로: 디버 태그 슬롯을 프로모션을 할 때 을 방출합니다.`mir.dbg_value`("이 소스 변수는 여기 이 SSA 값을 가지고 있습니다")`llvm.dbg.value`현재`full`계층이 실행되지 않습니다`mem2reg`그래서 평소에 지역 주민들은`dbg.declare`.

`ReferencePropagation`내부 로 포인터 할당을 대체할 수 있습니다.`AssignRef`이벤트:

```text
ptr = &slice[i] -> AssignRef(ptr, slice[i])
```

풀 모드는 이 이벤트를 안정적인 MIR에 전송한다. 지원되는 빌린 슬라이스 인덱스 형식에 대해서는 같은 프로그램 포인트에서 변수의 디버거 스택 슬롯에 주소를 입력한다:

```text
AssignRef -> address of slice[i] -> ptr's debugger slot
```

이입자는 전체 이벤트 세트를 먼저 검증합니다. 유효하지 않거나 지원되지 않는 이벤트는 번역 전에 그 목적지를 생략하므로 부분 기기가 발산되지 않습니다. 기본, 라인 테이블 및 릴리스 빌드는 변경되지 않습니다.. 프로모션 구제는 미래의 * 최적화 된 * 디버그 레벨의 기초 작업입니다.

### 변수 범위 및 인라인

변수 메타데이터에는 한 가지 추가 규칙이 있습니다.`arg: 1`이 디버그 스코프에 의해 이름 붙여진 함수의 공식 논증 #1을 의미합니다. " MIR 인라인링 후, 한 커널 몸의 두 변수는 둘 다 정직하게 `arg: 1`

```text
caller kernel:  data  -> arg #1 of caller
inlined helper: self  -> arg #1 of helper
```

그래서 cuda-oxide는 번역된 함수와 함께 rustc의 MIR 소스 스코프 테이블을 가지고 있습니다. 로컬 디버그 레코드는 소스-스코프 id만을 저장합니다. LLVM 수출자는 해당 id를 올바른 DWARF 형태로 변환합니다:

```text
kernel DISubprogram
  |
  +-- helper DISubprogram
        |
        +-- DILocation(..., inlinedAt: kernel callsite)
```

이게 왜냐구요`data`그리고 `self`둘 다 할 수 있어요`arg: 1`LLVM가 모듈을 모순이라고 거부하지 않고

현재 위치가 보수적입니다.`mem2reg`PHI 값처럼 동작하는 블록 논리를 생성하면 cuda-oxide는 추진된 로드/스토어 소스 포인트에서 변수를 기록합니다. 컴파일러가 결합 값을 생성했기 때문에 아직 열망하는 블록 엔트리 디버그 레코드를 추가하지 않습니다:

```text
then -> merge(10)
else -> merge(20)

merge(%x):
  ... old load site becomes dbg.value(%x, "x") ...
```

이는 정확하지만 극히 정확하지는 않습니다. 미래 패스는 소스 위치와 사전 범위 규칙이 명확한 후에 합병 블록에서 일찍 `x`를 표시 할 수 있습니다.

한 가지 더 컴파일러 유지 관리자 세부 사항: `mir.dbg_value`는 디버그 전용이지만, 오늘날 플리론 내부에는 여전히 정상적인 오페랑드 사용이 있습니다. 따라서 cuda-oxide는 이 디버깅 구조 포인트 이후 플리롱 DCE를 실행하지 않습니다. 나중에 추가하면, DCE는 디브깅 사용을 비 의미적으로 취급해야 합니다. LLVM이 `dbg.value`를 처리하는 것과 마찬가지로.

수출자가 생성한 PHI 노드는 일반적으로 얻을 수 없습니다`!dbg`: 그들은 SSA 회계이며 사용자가 작성한 소스 작업이 아닙니다. 나중에 변수 메타 데이터는 PHI 값을 참조 할 수 있습니다.`dbg.value`하지만 PHI 자체는 cuda-gdb에서 깜짝 스톱이 되어서는 안 됩니다.

---

## 상징 이름 위생화

함수 이름은 여러 단계로 흐르며 각각의 제한을 적용합니다:

```text
rustc_public (FQDN)            helper_fn::cuda_oxide_device_<hash>_vecadd
  ↓ body.rs (:: → __)
dialect-mir                    helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ call.rs (:: → __)
LLVM dialect                   helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ llvm-export (strip prefix)
Textual LLVM IR                @vecadd
  ↓ llc
PTX                            vecadd
```

이 경로에서 세 가지 변환이 발생합니다:

1. **`::``__`** -- 둘 다`body.rs`(기능 정의) 및 `call.rs`
(콜 타겟) 는 유효한 pliron/LLVM 식별자를 생성하기 위해 이중 하부 표시로  경로 분리자를 대체합니다. 양쪽 모두 동일한 변환을 적용하기 때문에 정의와 호출 사이트가 일치합니다.

2. **장치 전사자 제거 ** -- `export.rs`는 예약된
`cuda_oxide_device_<hash>_`전두두엽 (그리고 전두엽 FQDN 상자 전두통)`#[device]`를 통해 함수 이름을`reserved_oxide_symbols::device_base_name`이 접두어 MIR 레벨 검출에 존재하지만 최종 LLVM IR, PTX 또는 LTOIR 출력에는 나타나지 않아야합니다.

3. **장치 외부 전사자 제거** -- 에 대해`#[device] unsafe extern "C"`
함수, `call.rs`는 `cuda_oxide_device_extern_<hash>_` 전자를 `reserved_oxide_symbols::device_extern_base_name`를 통해 제거하여 LLVM IR는 외부 LTOIR (예: CCCL 라이브러리) 에 의해 수출된 원래 기호 이름을 참조합니다.

```{note}
이 수동 청소는 프레임워크가 업그레이드 될 때 pliron의 `Legaliser`로 대체됩니다. `Legaliser`는 `::`로 `_` 변환 및 충돌 탐지를 체계적으로 처리합니다.
```

---

## PTX 세대

LLVM 방언이 텍스트로 수출된 후`.ll`파일, 마지막 단계는 호출입니다`llc`- LLVM의 정적 컴파일러 - PTX 어셈블리를 생성하기 위해:

```bash
llc -march=nvptx64 -mcpu=sm_90 kernel.ll -o kernel.ptx
```

### 목표물 선택

파이프라인 탐사선은 아래 순서로`llc`를 탐색한다. LLVM 21은 최소  이전 릴리스는 cuda-oxide가 방출하는 TMA / tcgen05 / WGMMA 내장 함수 서명들을 거부한다.

|우선 순위 | 출처                                                    | 참고문서|
| :------- | :-------------------------------------------------------- | :-------------------------------------------------------------------- |
|1위      | `$CUDA_OXIDE_LLC` (설정된 경우)                                | 호출자에서 제공된 오버라이드, 당신이 지향하는 어떤 바이너리|
|2번      | `llvm-tools`                         | `<sysroot>/lib/rustlib/<host>/bin/llc` (`rustup`를 통해 자동 설치)|
|3번      | `llc-23``PATH`                                        | 디스트로 / `apt.llvm.org`LLVM 23의 설치|
|4번      | `llc-22``PATH`                                        | 디스트로 / `apt.llvm.org`LLVM 22의 설치|
|5번      | `llc-21``PATH`                                        | 디스트로 / `apt.llvm.org`LLVM 21의 설치|
|6번      | `llc``PATH`                                           | 리포트 하락 상태만 보고 합니다. LLVM 21보다 오래되면 실행 시 거부 됩니다.|

도구 체인 (`nightly-2026-10-11`) 가 NVPTX를 활성화 한 LLVM 23를 타고, 그래서`rustup component add llvm-tools`PATH 탐사선에서`llc-23`/ `llc-22`/ `llc-21`기존 LLVM 설치가 있는 사용자들을 위한 재발로 유지된다. 만약 탐사선 중 어느 것도 성공하지 않으면 파이프라인이 명백한 오류로 실패한다.`CUDA_OXIDE_LLC=/path/to/llc`, 하지만 간단한 커널은 LLVM 20 이하에서 컴파일할 수 있는 유일한 것이지요.

선택된 목표가 물리적 GPU와 일치하지 않으면 CUDA 드라이버는 JIT-인 PTX를 로드 시간에 컴파일합니다. 드라이버가 번역하는 동안 첫 번째 런칭은 대략 30ms가 소요됩니다. 후속 런칭에는 캐시 된 바이너리가 사용됩니다. 실제로는 거의 눈에 띄지 않습니다. JIT는 빠르며 캐시는 실행 중 지속됩니다.

환경 변수 `CUDA_OXIDE_TARGET`는 특정 목표가 필요한 경우에 자동 탐지를 대체합니다. 예를 들어, `sm_100a`은 일반 `sm_100` 목표에서 사용할 수 없는 블랙웰 특수한 `tcgen05` 기능을 활성화합니다.

`cargo oxide run`백엔드의 기능 기반 기본 설정 위에 자동 탐지 두 번째 계층을 추가합니다.`--arch`아니`CUDA_OXIDE_TARGET`설정되면 CUDA 장치 0의 컴퓨팅 능력을 쿼리하고 백엔드로 전달하여 생성된 모듈이 로컬 GPU에 로드될 것을 보장합니다. 전체 우선 순위는 입니다.`--arch`> `CUDA_OXIDE_TARGET`> 호스트 CC (에 대해)`run`오직) > 백엔드 기능 기반 기본`cargo oxide build`그리고 `cargo oxide pipeline`의도적으로 호스트-CC 단계를 건너뛰어 크로스 컴파일에 사용할 수 있도록 합니다.

낮춰진 모듈이 CUDA libdevice를 호출하면 cuda-oxide는 자동으로 `llc`에서 NVVM 경로로 전환합니다. 타입 지정 또는 불투명한 포인터 NVVM IR를 선택하기 전에 동일한 우선 순위로 목표를 해결합니다. 명시적인 NVVM/LTOIR 명령어는 여전히 `--arch`을 필요로합니다. 일반 빌드가 libdevise가 필요하다는 것을 발견 할 때만 기능 기반 재구축이 적용됩니다.

```{note}
LLVM 21은 왜? 2차원 대량 TMA 부하 내재가`tma_copy`, `gemm_sol`, 그리고 `tcgen05_matmul`10개의 연산자를 얻었다`addrspace(7)`그리고 `cta_group`LLVM 21의 매개 변수`llc`이 버전은 으로 거부합니다.`Intrinsic has incorrect argument type!`LLVM 버전마다 별도의 내재 발산자를 유지하기보다는 최소 21을 설정합니다.
```

### `opt`를 선택하여 일치합니다

`llc`파이프라인이 실행하는 유일한 LLVM 바이너리는 아닙니다. 중간 최적화 단계가 필요합니다.`opt`, 그리고 둘 다 같은 LLVM 주류에서 온해야합니다. 그들을 혼합하는 것은이 순서가 존재하는 것을 방지하는 실패입니다.`opt`하나의 방출에서 IR를 방출할 수 있습니다.`llc`다른 사람으로부터는 받아들이지 않습니다.

그래서`llc`먼저 위의 표로 풀리고, 그 주파수는 에서 읽는다.`llc --version`- `opt`다음으로 일치하도록 선택됩니다:

|우선 순위 | 출처                                             | 참고문서|
| :------- | :------------------------------------------------- | :-------------------------------------------------------------------- |
|1위      | `$CUDA_OXIDE_OPT` (설정된 경우)                         | 항상 존중됩니다. 선택한 `llc`에 대한 주요 불일치는 조용히 수정되지 않습니다. 두 바이너리 이름을 진단 기록합니다.|
|2번      | `opt`선택된 사람들 옆에`llc`                  | LLVM 설치는 도구들을 옆에 두고 있다. 여전히 버전 검증: 일치하지 않는 형제는 거부되고, `llc`의 자체 버전이 읽을 수 없을 때만 검증되지 않은 것으로 받아들여진다.|
|3번      | `llvm-tools` `opt`                | `<sysroot>/lib/rustlib/<host>/bin/opt`. `llc`와 같은 메이저로 필터링됩니다.|
|4번      | `opt-22`/ `opt-21`/ `opt``PATH`              | `llc`와 같은 메이저로 필터링됩니다.|

같은 메이저 `opt`가 존재하지 않으면, 해상도는 모든 거부된 후보자를 진단적으로 이름을 기록합니다. 독립 API는 요청된 최적화를 엄격하게 취급하고 실패합니다. 레거시 rustc 경로는 최적화되지 않은 상태로 돌아갑니다.

단계 2 키는 어느 `llc`가 승리하든, 단독으로 `CUDA_OXIDE_LLC`을 고정하는 것이 보통 충분하기 때문에, 일치하는 `opt`는 보통 그 옆에 발견되며, 쌍이 디렉토리 간에 분할될 때만 `CUDA_OXIDE_OPT`를 설정하는 것이 필요합니다.

### 파이프라인이 찾는 다른 두 가지 입력

`llvm-link`이 코드는 libdevice에 디바이스 코드가 호출되고 PTX 경로는 IR 레벨 링크로 해당 호출을 해결할 때만 필요합니다. `libdevice.10.bc`더 자주 필요합니다: NVVM 경로는 커널이 libdevice에 호출하는지 여부에 관계없이 최종적으로 완료하는 모든 모듈에 추가합니다. 둘 다 자체 오버라이드를 가지고 있습니다:

|변수 | 선택 | 디스커버리 설정이 풀리면|
| :------- | :------ | :------------------- |
|`CUDA_OXIDE_LLVM_LINK` | `llvm-link`이진 | 4단계와 같은 `opt`- 선택된 사람들 옆에`llc`, 그 다음 sysroot, 그리고 에 버전 이름을`PATH`, 으로 필터링`llc`변수 *가 설정되면,`CUDA_OXIDE_OPT`이진수 두 개 모두 두 개의 진단을 가진다.`llc`|
|`CUDA_OXIDE_LIBDEVICE` | 이 파일의 비트 코드는 `libdevice.10.bc`입니다. 존재하지 않는 경로는 조용히 건너뛰어집니다. | `<root>/nvvm/libdevice/libdevice.10.bc`각 `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`|

두 개의 경로는 서로 다르게 반응합니다. 일반적인 빌드가 실패하지 않습니다: 백엔드의 경로 결정은 IR 레벨 링크가 사용할 수 없다는 것을 보고 NVVM 경로로 돌아갑니다. 목표 선택에 설명된 동일한 자동 스위치입니다. 독립 API의 `Linking::Libdevice`엄격하고 앞면이 실패하는 대신,`LibdeviceUnavailable`실종된 조각의 이름을 지정하는 오류: `libdevice.10.bc`또는 `llvm-link`선택된 을 공유합니다`llc`이쪽이 메이저입니다.

## 기존 NVVM IR에서 원자 조작

LLVM 수준 원자는 NVVM 입력에 있는 원자 명령어이다. GPU가 원자 하드웨어를 가지고 있는지에 대한 진술은 아니다. CUDA 12의 LLVM 7 NVVM 방언은 하위 집합만을 받아들인다:

|LLVM 구성 | CUDA 12 NVVM IR | 오늘 다-옥시드 레거시 경로|
|:----------------|:----------------|:-----------------------------|
|원자 화물 또는 저장소 | LLVM 원자 명령어로 지원되지 않습니다 | 거절됐고 또 내려야 돼|
|`fence` | LLVM `fence`는 지원되지 않습니다. NVVM 메모리 장벽 작업이 필요합니다. | 정밀한 순서/범위 지도가 나올 때까지 거절|
|`cmpxchg` | `i32`/`i64`, 더해`i128``compute_90+`; 글로벌/공유 지표 또는 거기에 참조하는 것으로 알려진 일반 지표 | 일반, 글로벌 및 공유 주소 공간에서 느슨한 (단조) 순서에서 `i32`/`i64`에 대해 네이티브로 유지됩니다. 느슨하거나 블록 / 시스템 범위보다 강한 성공 순서가 직선 PTX로 다시 작성됩니다. 다른 너비 (모든 기능에 포함된 `i128`), 다른 주소 공간 및 성공 순서보다 강한 실패 순서는 거부됩니다.|
|`atomicrmw` | 정수 `xchg`, `add`, `sub`, `and`, `or`, `xor`, `max`, `min`, `umax`, 그리고 `umin``i32`/`i64``i128 xchg``compute_90+`; 같은 주소 공간 제한 | 일반, 글로벌 및 공유 주소 공간에서 `i32`/`i64`에 대해 고유하게 유지됩니다. 그 열 개의 연산에 대해 단조로운 순서로 유지됩니다|
|NVVM 원자 내적성 | 부동 소수점 원자 추가 등 선택된 추가 작업을 제공 | 느긋하고 장치 스코프 `atomicrmw fadd``f32`/`f64`일반/전지적/공유 주소 공간에서 정확한 레거시 내장 함수으로 낮아집니다. 다른 * 부동 소수점 * 원자 RMW 작업은 여전히 거부됩니다|

일반적인 LLVM-to-PTX 경로는 기존의 원자 지원을 유지합니다. 위의 표는 레거시 NVVM 합법화자만을 설명합니다.

두 가지 규칙이 그 열을 설명합니다. 첫 번째는 타입, 주소 공간, 순서 및 범위가 살아남을 수 없다는 것을 증명할 수 없는 것이 방출되지 않는다는 것입니다: [# 921]](https://github.com/NVlabs/cuda-oxide/pull/921) 및 [#923]](https://github.com/NVlabs/cuda-oxide/pull/923)는 정수 RMW를 허용하고 그 특성을 증명하는 대신 비교 교환을 통해 그 특성에 대해 완화시키는 것이 아니라, 생존 거부가 증명이 유효하지 않은 경우입니다.

두 번째는 ** 범위는 결코 적합하도록 약화되지 않습니다. libNVVM의 레거시 방언은 표현할 수 없는 것을 떨어 뜨립니다: 순서형 `cmpxchg`는 순서 형식 및 주변 울타리가없는 맨한 `atom.cas.b32`/`.b64`로 낮아집니다. 그래서 블록 또는 시스템 스코프 된 정수 원자 및 성공 순서가 느슨한 것보다 더 강한 비교 교환은 libNVVM에 전달되는 대신 범위와 순서를 명시하는 직선 PTX로 다시 작성됩니다. 그 다시 작성하려면 `sm_70` 또는 더 새가 필요합니다. 성공 순서보다 더 강한 실패 순서는 결코 다시 작성되지 않습니다. 그것은 완전히 거부됩니다.

단조로운 것은 유일한 주문입니다`atomicrmw`MIR를 하위 변환 것은 이미 강렬한 명령들을 분리된 배열로 나누었기 때문에 명령 자체가 아무것도 가지고 있지 않습니다.

레거시 사양은 또한 `cmpxchg`의 `weak` 마커와 실패 순서를 받아들이지만 무시합니다.

정확한 허용된 유형 및 작업을 위해 [CUDA 12.4 NVVM IR 사양]](https://docs.nvidia.com/cuda/archive/12.4.0/nvvm-ir-spec/index.html)를 참조하십시오.

---

## 모든 걸 합쳐놓는 것

여기 `lower_mir_to_llvm`가 모듈을 처리할 때 일어나는 모든 일련의 사건들이 있습니다:

```text
1. LLVM dialect types and operations are registered automatically (link-time)
2. For each MirFuncOp in the module:
   a. Create `llvm.func` with flattened type signature
   b. inline_region: move `dialect-mir` blocks into the LLVM function
   c. Build entry prologue (reconstruct aggregates from flat args)
   d. Run DialectConversion:
      ├── Walk every `dialect-mir`/`dialect-nvvm` op (def-before-use order)
      ├── Invoke MirToLlvmConversion::rewrite for each op
      ├── Converter emits LLVM dialect op(s) via DialectConversionRewriter
      └── Framework patches block arg types automatically
3. Export the LLVM dialect to textual LLVM IR (.ll) (with PHI node conversion)
4. Invoke llc to produce .ptx
```

4단계 후, 당신은 을 가지고 있습니다.`.ptx`CUDA 드라이버가 로드하고 실행할 수 있는 파일입니다.`mir.checked_add``add.s32`완료되었습니다.

---

낮출 파이프 라인은  맛의 IR를 GPU 준비 된 LLVM IR로 변환합니다. 새로운 GPU 연산을 추가하는 방법에 대한 실제 설명은 [새로운 내성 추가] ](adding-new-intrinsics.md) 참조하십시오.

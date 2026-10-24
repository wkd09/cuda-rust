# 커널 계열

GPU 커널은 종종 다른 문제 형태, 하드웨어 기능, 자원 제한 또는 튜닝 결정에 특화되어 있습니다. 따라서 단일 논리적 연산은 여러 가지 미리 컴파일된 진입점를 가질 수 있습니다.

`KernelFamily`는 컴파일된 구현들의 집합을 나타내고 검증된 호스트 측 선택 메커니즘을 제공합니다.

이 구분은 두 가지 독립적인 결정을 구분합니다.

```text
Eligibility: can this variant safely execute this problem?
Preference:  which eligible variant should execute?
```

이 구별은 중요합니다. 성능 정책은 실행, 형상, 하드웨어 또는 자원 요구 사항을 위반하는 변형을 결코 선택해서는 안됩니다.

`KernelFamily`CPU에서 완전히 선택을 수행합니다. 커널을 컴파일하거나 CUDA 장치를 검색하거나 항목을 벤치마크하거나 스레드를 생성하거나 파일 시스템에 액세스하지 않습니다. 응용 프로그램은 모든 관련 문제 및 하드웨어 사실을 제공합니다.

## 커널 계열를 사용할 때

작업이 이미 컴파일된 구현의 작은 고정 집합을 가지고 있을 때 `KernelFamily`를 사용한다. 예를 들어:

* 다른 출력 타일 크기
* 다른 블록 크기
* 특정 컴퓨팅 기능을 위한 특화된 텐서 코어 커널
* 작은 입력과 큰 입력 구현
* 서로 다른 공유 메모리 요구 사항이 있는 커널
* 서로 다른 문제 형상 범위에 맞춰진 커널
* 일반 재구성 및 하나 이상의 최적화된 변형

커널 계열는 런타임 레지스트리나 JIT 컴파일러가 아닙니다. 그 변수 수는 일반 const이며 따라서 Rust 타입의 일부입니다:

```rust
KernelFamily<Id, Entry, Meta, const N: usize>
```

이 가족은 정확히 `N` 변종을 소유하고 있습니다.

이 제한된 표현은 사용할 수 있는 커널이 빌드 시점에 결정될 때 유용하며 명시적이고 감사될 수 있어야 합니다.

## 선택 모델

선택 흐름은 두 가지 모드를 가지고 있습니다:

```text
Force(id) -> resolve variant -> validate ----------------------> Override

Auto      -> cache lookup -> resolve -> validate --------------> Cache
          -> filter eligible variants
          -> selector chooses an ID
          -> validate selector output
          -> cache selected ID --------------------------------> Selector
```

`SelectionMode::Force(id)`는 검증된 수동 오버라이드입니다. 캐시와 선택기를 우회하지만 적정성 검증을 우회하지 않습니다.

`SelectionMode::Auto`는 캐시된 안정적인 변수 ID를 사용하려고 시도합니다. 알려지지 않은 또는 새로 지원되지 않는 캐시 항목은 미수로 취급됩니다. 선택자는 지원 가능한 변수만 볼 수 있습니다.

모든 성공한 선택은 그 출처를 보고합니다:

```rust
pub enum SelectionSource {
    Override,
    Cache,
    Selector,
}
```

이것은 응용 프로그램에서 운영자 오버라이드, 캐시 결정 및 로그 및 진단에서 새로운 선택자 결정을 구별 할 수 있습니다.

## 가족 해부학

가족이란 안정적인 가족 정체성과 다양한 형태를 가진다.

```rust
pub struct KernelFamily<Id, Entry, Meta, const N: usize> {
    // Private fields
}
```

각 변수는 호출자 정의 값 3개를 포함합니다:

```rust
pub struct KernelVariant<Id, Entry, Meta> {
    // Private fields
}
```

|구성 요소 | 목적|
| :-------- | :----------------------------------------------------- |
|`Id`      | 변수에 대한 안정적인 의미 정체성|
|`Entry`   | 변종과 관련된 호출 가능한 입력 또는 발사체|
|`Meta`    | 지원 및 선호도 메타데이터|
|`N`       | 컴파일 시간 변수 수|

`Entry`는 의도적으로 일반화되어 있습니다. 함수 포인터, 타입 지정 런칭 래퍼, 생성된 메소드를 설명하는 엔룸 또는 다른 응용 프로그램 정의 값이 될 수 있습니다.

`Meta`또한 응용 프로그램 정의입니다. 일반적으로 요구 사항과 조정 사실을 저장합니다.

* 최소 컴퓨팅 능력
* 타일 크기
* 정렬 요구 사항
* 공유 메모리 사용
* 지원되는 데이터 타입
* 문제 크기의 범위가 선호되는 문제
* 발사 기하학
* 조율 점수 또는 우선 순위

## 안정적인 변수 ID를 정의합니다

모든 변종은 명시적인 ID가 필요합니다:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    Scalar,
    Chunked,
}
```

ID는 다음과 같이 사용됩니다:

* 수동 오버라이드
* 캐시
* 선택기
* 진단
* 구성 파일
* 환경 변수

ID는 컴파일된 변수의 의미적 정체성을 나타낼 수 있어야 합니다.

좋은 ID는 엔룸 값이나 안정 문자열을 포함합니다. 그 의미는 응용 프로그램에서 제어됩니다.

```rust
enum GemmVariantId {
    M256xN256,
    M512xN256,
}
```

빌드나 프로세스 사이에서 정체성이 바뀔 수 있는 값을 사용하지 마십시오:

* 배열 지수
* 지표
* `TypeId`
* 포맷된 디버그 출력
* 생성 함수의 주소
* 알고리즘이나 입력값이 안정적이지 않은 해시

배열 위치는 안정적인 ID를 대체할 수 없습니다. 선언 순서는 선택 정책에 영향을 줄 수 있지만 캐시 정체성을 정의해서는 안됩니다.

## 가족 건설

`KernelVariant::new`로 변형을 만들:

```rust
let variant = KernelVariant::new(
    VariantId::Chunked,
    chunked_entry,
    VariantMetadata {
        min_len: 1024,
        chunk_width: 256,
    },
);
```

이 가족들을 만들려면 `KernelFamily::try_new`를 사용해야 합니다:

```rust
let family = KernelFamily::try_new(
    "vector-copy",
    1,
    [
        scalar_variant,
        chunked_variant,
    ],
)?;
```

그 주장은 다음과 같습니다:

```rust
KernelFamily::try_new(
    name: &'static str,
    revision: u32,
    variants: [KernelVariant<Id, Entry, Meta>; N],
)
```

건설은 세 가지 구조적 변수를 검증합니다:

1. 이 가족에는 적어도 하나의 변종이 있습니다.
2. 가문은 빈 공간이나 빈 공간만 있는 게 아닙니다.
3. 변종 ID는 가족 내에서 독특합니다.

건축은 돌아 갈 수 있습니다:

```rust
pub enum KernelFamilyBuildError {
    EmptyFamily,
    EmptyFamilyName,
    DuplicateVariantId {
        first_index: usize,
        duplicate_index: usize,
    },
}
```

## 가족 정체성 및 개정

가족 이름과 개정판은 선택 및 캐시 네임스페이스:

```rust
let id = family.id();

println!("{}", id.name());
println!("{}", id.revision());
println!("{id}"); // vector-copy@1
```

영구 또는 메모리 내 캐시는 두 값 모두 키에 포함되어야 합니다.

기존의 캐시된 결정이 더 이상 같은 것을 의미하지 않을 때 수정을 증가하십시오.

수정 변경은 다음을 수정할 때 필요합니다.

* 변종 회원
* 변수 ID의 의미
* 지원자격 규정
* 선택자 또는 선호 정책
* 정렬 방법
* 선택에 의해 사용된 메타데이터
* 지정 명령, 선택 및 조정이 명시적으로 명시되지 않는 한
주문에 관계없이

예를 들어:

```rust
KernelFamily::try_new("vector-copy", 1, variants_v1)?;
KernelFamily::try_new("vector-copy", 2, variants_v2)?;
```

개정법의 변경은 첫 계약에 따라 결정된 결정을 두 번째 계약에 의해 해석되는 것을 방지합니다.

가족 이름은 논리적 동작과 전문화 축을 설명해야 합니다:

```text
image-filter/radius
gemm/output-tile
reduction/block-size
attention/head-dimension
```

일시적인 구현 위치 또는 프로세스 특정 상태에 기반한 이름을 피하십시오.

## `KernelProblem`와 함께 지원 가능

지원 여부는 특정 변종이 문제를 안전하게 처리할 수 있는지 여부에 대한 답입니다.

실행:

```rust
pub trait KernelProblem<Variant> {
    type Rejection: std::error::Error + Send + Sync + 'static;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection>;
}
```

문제 유형은 발사 여부를 확인하는 데 필요한 모든 사실을 포함해야 합니다.

예를 들어:

```rust
#[derive(Clone, Copy, Debug)]
struct CopyProblem {
    len: usize,
}
```

검증은 다음을 고려할 수 있습니다:

* 문제 차원
* 분할성 요구사항
* 버퍼 정렬
* 데이터 타입
* 장치 계산 능력
* 공유 메모리 용량
* 네트워크 및 블록 제한
* 협동조합 발사 지원
* 클러스터 발사 지원
* 정수 변환 및 할당 넘치는
* 응용 프로그램 특유의 의미적 제약

구현은 결정적이고 부작용이 없어야 합니다.

적격성은 안전성과 정확성의 경계입니다. 그것은 성능 선호도를만 암호화해서는 안됩니다.

예를 들어, 256로 나눌 수 있는 길이가 필요한 변수는 다른 계층이 선택하지 않을 것으로 예상되더라도 호환되지 않는 문제를 거부해야합니다.

## `KernelSelector`와 선호

선택자는 이미 적정성 검증을 통과 한 변수 중 하나를 선택합니다:

```rust
pub trait KernelSelector<Problem, Variant, Id> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn select(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        eligible: &[&Variant],
    ) -> Result<Id, Self::Error>;
}
```

`eligible`슬라이스는 두 가지 보장이 있습니다:

1. 빈 곳이 아닙니다.
2. 가족 신고 순서를 유지합니다.

선택자는 안정적인 변수 ID를 반환하지만 배열 인덱스 또는 입력 참조는 반환하지 않습니다.

선택자는 다음과 같은 정책을 실행할 수 있습니다.

* 문제에서 지원하는 가장 큰 타일을 선택
* 임계치 아래의 작은 입력 커널을 선택
* 오프라인 벤치마크 결과를 사용하여 선택
* 하드웨어 생성을 사용해서 선택
* 첫 번째 지원 가능한 변형을 선택
* 응용 프로그램 구성에서 선택
* 외부 튜닝 서비스를 호출

선택자는 제공된 적정 슬라이스에서 ID를 반환해야 합니다. `KernelFamily`는 이 요구 사항을 확인하고 유효하지 않은 선택자 출력을 거부합니다.

클로저는 그 서명이 속성과 일치할 때 선택자로 작용할 수도 있다. 모든 클로저는 매개 변수 타입을 주석한다. 주석 없이 클로져 라이프타임 인퍼런스는 한 특정 호출에 빌려서 고정시킬 수 있으며 rustc는 "`FnMut`의 구현이 충분히 일반적이지 않다"는 채로 클로서를 거부한다. 실패할 수 없는 선택자는 `std::convert::Infallible`를 오류 타입으로 사용할 수 있다:

```rust
let mut selector =
    |_: KernelFamilyId, _: &CopyProblem, eligible: &[&Variant]| {
        Ok::<_, Infallible>(*eligible[0].id())
    };
```

## 자동 선택

자동 선택 사용:

```rust
SelectionMode::Auto
```

전체 호출은:

```rust
let selected = family.select(
    &problem,
    SelectionMode::Auto,
    &mut selector,
    &mut cache,
)?;
```

자동 경로는 다음 동작을 수행합니다:

1. 캐시에서 안정적인 변종 ID를 구해보세요.
2. 가족 내에서 캐시된 ID를 해결하세요.
3. 현재 문제로 캐시된 변형을 재확인합니다.
4. 유효한 캐시 타겟을 즉시 반환합니다.
5. 그렇지 않으면 가족 구성원 모두를 확인하세요.
6. 신청할 수 있는 변종을 신고 순서대로 유지합니다.
7. 변종이 없는 경우 실패합니다.
8. 선택자에게 지원 가능한 조각 중에서 선택하도록 요청합니다.
9. 선택자가 가족 구성원을 반환했는지 확인합니다.
10. 반환된 구성원이 지원 가능한 슬라이스에 있는지 확인합니다.
11. 확인된 선택된 ID를 캐시에 저장합니다.
12. 선택된 변수를 `SelectionSource::Selector`로 반환합니다.

캐시된 아이디는 단서가 아니라 권한입니다.

캐시 항목은 다음과 같은 이유로 유효하지 않을 수 있습니다.

* 캐시는 이전 응용 프로그램 버전으로 작성되었습니다
* 지원 여부는 하드웨어에 달려 있습니다
* 문제 제약이 바뀌었다
* 영구 캐시가 손상되었습니다
* 변종이 제거되었습니다
* 이전 정책이 다른 수정법을 잘못 사용했네요

이러한 이유로, 모든 캐시된 ID는 사용 전에 해결되고 검증됩니다.

## 강제선택

사용:

```rust
SelectionMode::Force(id)
```

전화자가 변형을 명시적으로 요청할 때

대표적인 자료는:

* 명령 줄 옵션
* 환경 변수
* 시험 구성
* 디버깅 컨트롤
* 성능 실험
* 사업자 정책

예제:

```rust
let selected = family.select(
    &problem,
    SelectionMode::Force(VariantId::Scalar),
    &mut selector,
    &mut cache,
)?;
```

강제선택:

* 캐시 검색을 우회합니다
* 선택자 호출을 우회합니다
* 캐시를 업데이트하지 않습니다
* 요청된 ID가 가족에 속하는지 확인합니다
* 요청된 변종이 지원 가능하다는 것을 확인합니다

알 수 없는 강제 ID가 `UnknownForcedVariant`를 반환합니다.

알려진 하지만 호환되지 않는 변수가 `IneligibleForcedVariant`를 반환합니다.

따라서 수동 오버라이드는 발사 요구 사항을 조용히 우회할 수 없습니다.

## 선택 결과

성공적인 호출은 빌린 `SelectedVariant`를 반환합니다:

```rust
pub struct SelectedVariant<'family, Id, Entry, Meta> {
    // Private fields
}
```

변종과 출처에 액세스 할 수 있습니다:

```rust
let variant = selected.variant();
let source = selected.source();

println!("selected {:?} from {}", variant.id(), source);
```

선택된 값은 가족에서 빌려옵니다. 그것은 변형 항목이나 메타 데이터를 복사하거나 이동하지 않습니다.

변종은 다음과 같습니다:

```rust
variant.id();
variant.entry();
variant.metadata();
```

`SelectionSource`는 `Display`를 구현합니다:

```text
override
cache
selector
```

## 전체 예제

다음 예제에서는 일반 CPU 함수를 항목으로 사용합니다. 항목이 컴파일된 CUDA 커널 주위의 래퍼로 입력될 때 동일한 선택 구조가 적용됩니다.

```rust
use cuda_host::{
    KernelFamily, KernelFamilyBuildError, KernelFamilyId, KernelProblem, KernelSelector,
    KernelVariant, NoKernelSelectionCache, SelectionMode, SelectionSource,
};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    Scalar,
    Chunked,
}

type Entry = fn(&[f32], &mut [f32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VariantMetadata {
    min_len: usize,
    chunk_width: usize,
}

type Variant = KernelVariant<VariantId, Entry, VariantMetadata>;
type CopyFamily = KernelFamily<VariantId, Entry, VariantMetadata, 2>;

fn scalar_copy(input: &[f32], output: &mut [f32]) {
    output.copy_from_slice(input);
}

fn chunked_copy(input: &[f32], output: &mut [f32]) {
    const WIDTH: usize = 256;

    for (input_chunk, output_chunk) in input
        .chunks_exact(WIDTH)
        .zip(output.chunks_exact_mut(WIDTH))
    {
        output_chunk.copy_from_slice(input_chunk);
    }
}

fn copy_family() -> Result<CopyFamily, KernelFamilyBuildError> {
    KernelFamily::try_new(
        "book/vector-copy",
        1,
        [
            KernelVariant::new(
                VariantId::Scalar,
                scalar_copy as Entry,
                VariantMetadata {
                    min_len: 0,
                    chunk_width: 1,
                },
            ),
            KernelVariant::new(
                VariantId::Chunked,
                chunked_copy as Entry,
                VariantMetadata {
                    min_len: 1024,
                    chunk_width: 256,
                },
            ),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
struct CopyProblem {
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyVariantRejection {
    variant: VariantId,
    required_min_len: usize,
    required_multiple: usize,
    actual_len: usize,
}

impl fmt::Display for CopyVariantRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} requires len >= {} and len divisible by {}, but len is {}",
            self.variant,
            self.required_min_len,
            self.required_multiple,
            self.actual_len,
        )
    }
}

impl Error for CopyVariantRejection {}

impl KernelProblem<Variant> for CopyProblem {
    type Rejection = CopyVariantRejection;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection> {
        let metadata = variant.metadata();

        if self.len < metadata.min_len
            || !self.len.is_multiple_of(metadata.chunk_width)
        {
            return Err(CopyVariantRejection {
                variant: *variant.id(),
                required_min_len: metadata.min_len,
                required_multiple: metadata.chunk_width,
                actual_len: self.len,
            });
        }

        Ok(())
    }
}

struct CopySelector;

impl KernelSelector<CopyProblem, Variant, VariantId> for CopySelector {
    type Error = Infallible;

    fn select(
        &mut self,
        _family: KernelFamilyId,
        _problem: &CopyProblem,
        eligible: &[&Variant],
    ) -> Result<VariantId, Self::Error> {
        let preferred = eligible
            .iter()
            .copied()
            .max_by_key(|variant| variant.metadata().chunk_width)
            .expect("eligible is guaranteed to be non-empty");

        Ok(*preferred.id())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let family = copy_family()?;

    let input = vec![1.0f32; 4096];
    let mut output = vec![0.0f32; input.len()];
    let problem = CopyProblem { len: input.len() };

    let mut selector = CopySelector;
    let mut cache = NoKernelSelectionCache;

    let selected = family.select(
        &problem,
        SelectionMode::Auto,
        &mut selector,
        &mut cache,
    )?;

    println!(
        "selected {:?} from {}",
        selected.variant().id(),
        selected.source(),
    );

    assert_eq!(selected.variant().id(), &VariantId::Chunked);
    assert_eq!(selected.source(), SelectionSource::Selector);

    let entry = *selected.variant().entry();
    entry(&input, &mut output);

    assert_eq!(input, output);

    let forced = family.select(
        &problem,
        SelectionMode::Force(VariantId::Scalar),
        &mut selector,
        &mut cache,
    )?;

    assert_eq!(forced.variant().id(), &VariantId::Scalar);
    assert_eq!(forced.source(), SelectionSource::Override);

    Ok(())
}
```

입력 길이가 4096인 경우 두 가지 변종 모두 지원됩니다. 선택자는 더 큰 `chunk_width`가 있는 변종을 선호합니다.

1000의 입력 길이의 경우, 오직 `Scalar`만이 자격이 있습니다. 왜냐하면 `Chunked`는 적어도 1024개의 요소와 256로 나눌 수 있는 길이가 필요하기 때문입니다.

선택 메커니즘은 `chunk_width`의 의미를 알지 못합니다. 자격 및 선호 시맨틱스는 전적으로 응용 프로그램에 속합니다.

## 캐시를 추가합니다

선택이 실행될 때마다 `NoKernelSelectionCache`를 사용한다:

```rust
let mut cache = NoKernelSelectionCache;
```

반복된 문제 클래스에서는 `KernelSelectionCache`를 구현합니다:

```rust
pub trait KernelSelectionCache<Problem, Id> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
    ) -> Result<Option<Id>, Self::Error>;

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        variant: &Id,
    ) -> Result<(), Self::Error>;
}
```

이전 예제에서 간단한 in-memory 캐시는 문제 길이의 키의 일부로 사용할 수 있습니다:

```rust
use cuda_host::{KernelFamilyId, KernelSelectionCache};
use std::collections::HashMap;
use std::convert::Infallible;

#[derive(Default)]
struct MemoryCache {
    values: HashMap<(KernelFamilyId, usize), VariantId>,
}

impl KernelSelectionCache<CopyProblem, VariantId> for MemoryCache {
    type Error = Infallible;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &CopyProblem,
    ) -> Result<Option<VariantId>, Self::Error> {
        Ok(self.values.get(&(family, problem.len)).copied())
    }

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &CopyProblem,
        variant: &VariantId,
    ) -> Result<(), Self::Error> {
        self.values.insert((family, problem.len), *variant);
        Ok(())
    }
}
```

응용 프로그램은 어떤 문제 속성이 캐시 동등성을 정의하는지 결정합니다.

예를 들어, GEMM 캐시 키에는 다음이 포함될 수 있다:

```text
family name
family revision
compute capability
M
N
K
input type
output type
layout
alignment class
```

유치권이나 선호도를 바꿀 수 있는 부동산을 빼놓지 마세요.

캐시 구현은 다음과 같습니다:

* 메모리 내의 지도
* 프로세스 전체 캐시
* 데이터베이스
* 구성 파일
* 기준 결과 저장소
* 원격 조정 서비스

캐시 오류는 명시적입니다.`lookup`또는`store`오류를 반환하고 선택 중지 및 반환`CacheLookupFailed`또는`CacheStoreFailed`.

최대의 노력 캐시는 내부적으로 저장 실패를 놓친 또는 그 정책이 적절할 때 저장하지 않는 것으로 변환 할 수 있습니다.

## 오류 처리

`KernelFamily::select`네 계층에서 오류를 보고할 수 있습니다:

1. 강제선택
2. 지원자격 검증
3. 선택기 실행 또는 출력 검증
4. 캐시 작업

주요 변종은:

|오류                               | 의미|
| :---------------------------------- | :---------------------------------------------------------- |
|`UnknownForcedVariant`              | 강제 신분증은 가족 구성원이 아닙니다|
|`IneligibleForcedVariant`           | 강제적인 가족 구성원은 문제를 처리할 수 없습니다|
|`NoEligibleVariants`                | 가족 구성원 모두가 검증에 실패했습니다|
|`SelectorFailed`                    | 선택자가 자신의 정책 오류를 반환합니다|
|`SelectorReturnedUnknownVariant`    | 선택자는 가족 외부의 ID를 반환합니다|
|`SelectorReturnedIneligibleVariant` | 선택자는 자격이 없는 가족원을 돌려주었습니다|
|`CacheLookupFailed`                 | 캐시 검색에서 오류가 발생했습니다|
|`CacheStoreFailed`                  | 캐시 저장소 오류가 발생했습니다|

알려지지 않거나 적합하지 않은 캐시 ID는 선택 오류가 아닙니다. 캐시 내용은 신뢰할 수없는 힌트이기 때문에 캐시 미스로 취급됩니다.

선택 오류는 선택자가 현재 선택 호출 중에 권한 있는 적정 슬라이스를 수신하기 때문에 오류입니다.

## 신고명령

가족 변종은 신고 순서를 유지합니다:

```rust
KernelFamily::try_new(
    "operation",
    1,
    [
        fallback,
        medium,
        aggressive,
    ],
)?;
```

선택자는 동일한 상대 순서로 지원 가능한 변종을 수신합니다.

이것은 다음과 같은 결정적 정책을 지원합니다:

```rust
Ok(*eligible[0].id())
```

또는 우선 순위 지정 순서로 암호화됩니다.

선언 순서가 선택이나 조정에 영향을 주면 순서를 변경하면 가족 수정 부름이 필요합니다.

정책이 단순한 순서로 된 재발보다 더 복잡할 때 명시적인 메타데이터를 선호합니다. 메타 데이터는 정책을 검사하고 테스트하고 진화하는 것을 더 쉽게합니다.

## 가족들을 시험해보고

커널 계열 정책은 호스트 측 논리이며 CUDA 장치 없이 테스트할 수 있다.

테스트는 적어도 다음을 확인해야 합니다:

* 빈 가정은 거절됩니다
* 빈 이름들은 거절됩니다
* 복제된 ID는 거부됩니다
* 강제 선택은 선택기와 캐시를 우회합니다
* 강제선택은 여전히 자격을 확인합니다
* 유효한 캐시 타겟은 선택자를 우회합니다
* 알려지지 않은 캐시 ID는 선택자에 다시 떨어집니다
* 새로 지원되지 않는 캐시 ID는 선택자에 다시 떨어집니다
* 구식 캐시 항목이 대체됩니다
* 선택자는 지원 가능한 변종만 받습니다
* 신고명령은 유지됩니다
* 불명확 선택기 출력이 거부됩니다
* 지원되지 않는 선택기 출력이 거부됩니다
* 지원 가능한 변수가 없는 경우 오류가 발생합니다
* 캐시 검색 실패가 퍼져나갑니다
* 캐시 저장장 오류가 퍼져나갑니다
* 가족 수정 은 캐시 된 결정을 고립 합니다

리포지토리 통합 테스트는 실행 가능한 예를 제공합니다:

```text
cuda-oxide/crates/cuda-host/tests/kernel_family.rs
```

으로 실행하세요

```text
cargo test -p cuda-host --test kernel_family
```

## 생성된 CUDA 발사기를 사용해서

실제 적용에서는,`Entry`일반적으로 콘크리트형 발사체를 나타냅니다.

예를 들어:

```rust
type KernelLauncher = unsafe fn(
    &kernels::LoadedModule,
    &CudaStream,
    LaunchConfig,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
) -> Result<(), cuda_core::DriverError>;
```

그 다음 가족들은 컴파일된 전문성당 1개의 발사기를 저장합니다:

```rust
type Family =
    KernelFamily<VariantId, KernelLauncher, VariantMetadata, 2>;
```

선택은 발사기와 메타데이터를 반환합니다:

```rust
let selected = family.select(
    &problem,
    mode,
    &mut selector,
    &mut cache,
)?;

let launch = *selected.variant().entry();
let metadata = selected.variant().metadata();

println!(
    "launching {:?} from {}",
    selected.variant().id(),
    selected.source(),
);

unsafe {
    launch(
        &module,
        &stream,
        launch_config,
        &input,
        &mut output,
    )?;
}
```

`KernelFamily`는 변종 선택을 검증합니다. 그렇지 않으면 unsafe 원시 발사를 안전하게 만들지 않습니다.

실행 기하학, 포인터 유효성, 버퍼 크기, 컨텍스트 정체성, 커널 ABI 의무는 여전히 실행 API에 속한다. 커널 기하학을 정적으로 표현할 수 있는 경우 준비된 실행 계약을 선호한다. [실행 커널]](launching-kernels.md) 참조.

## 실제 사례

`gemm_sol_final`예를 들어 두 Blackwell GEMM 출력 틸 전문에 대해 두 항목 가족을 사용합니다:

```text
M256xN256
M512xN256
```

그 가족 메타 데이터에는 다음이 포함됩니다:

* 커널 이름
* 출력 타일
* 타일 크기 M
* 타일 크기 N
* K 타일 복수
* 문제 크기의 우선순위 기준

적격성 검사는 행렬의 크기, 타일 나눌 수 있는 정도, 정수 범위, 할당 크기의 오버플로우 및 CUDA 그리드 제한을 검증합니다.

선택자는 가장 큰 선호 타일을 선택합니다. 그 임계값이 문제와 호환됩니다.

환경 변수 `GEMM_SOL_VARIANT`는 다음과 같이 나타냅니다:

```text
auto
m256xn256
m512xn256
```

강제 값은 유효한 오버라이드 값으로 유지됩니다. 컴파일된 형상이 행렬 차량과 호환되지 않는 타일을 선택할 수 없습니다.

본문:

* [`gemm_sol_final`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol_final)
* [`KernelFamily` 통합 테스트](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/crates/cuda-host/tests/kernel_family.rs)
* [`cuda-host` 커널 계열 구현](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/crates/cuda-host/src/kernel_family.rs)
* [`cuda-host` README](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/crates/cuda-host/README.md)

## 설계 지침

새로운 가족을 소개할 때 다음 규칙을 사용하세요.

### 가족들끼리 한정된 생활을 해

의미 있는 컴파일된 변수들의 작은 집합을 선호합니다.

큰 동적으로 변화하는 세트는 일반적으로 레지스트리, 생성 된 배포 테이블 또는 튜닝 데이터베이스가 더 나은 추상화임을 나타냅니다.

### ID를 의미적으로 만들자

ID는 어떤 컴파일된 계약이 선택되었는지 설명해야 하며, 배열에 나타나는 장소가 아닌 것입니다.

### 적정성 정립

안전하거나 올바른 실행에 필요한 모든 조건은 `KernelProblem::validate`에 속한다.

unsafe 변종을 피하기 위해 선택기에 의존하지 마십시오.

### 성능을 우선시하세요

선택자는 이미 유효한 선택 중 하나를 선택해야 합니다.

이것은 안전 검증으로부터 독립된 조정 정책을 유지합니다.

### 캐시 항목을 신뢰할 수 없는 것으로 처리합니다

항상 `KernelFamily`가 캐시된 ID를 해결하고 재확인하도록 허용합니다.

가족 확인 없이 캐시된 ID에서 직접 시작하지 마세요.

### 버전 정책 변경

가족 개정을 증가하면 캐시 된 선택은 새로운 코드에서 다른 의미를 가질 수 있습니다.

### 관찰 가능성 유지

`SelectionSource`, 가족 정체성, 선택된 ID 및 진단에 관련된 문제 사실을 기록하십시오.

이것은 특히 연산자 오버라이드, 캐시 조정 및 현재 선택자 행동을 비교할 때 유용합니다.

## 요약

`KernelFamily`는 시간보다 앞서 컴파일된 커널 전문화를 위한 제한되고 검증된 디스파치 레이어를 제공합니다.

그 핵심 속성은:

* 일정한 가족 구성원
* 안정적인 의미 변수 ID
* 호출자 정의 항목 및 메타 데이터
* 명시적 지원성 검증
* 독립적 인 선호 선택
* 유효한 수동 오버라이드
* 재확인된 캐시 항목
* 명백한 가족 개정
* 관찰 가능한 선택의 출처
* CPU만 사용하며 장치와 독립된 정책 논리

추상화는 커널의 튜닝 방식을 결정하지 않습니다. 그것은 튜팅 정책을 명시적이고 테스트 가능하며 컴파일된 CUDA 런처와 통합할 수 있도록 필요한 변수를 제공합니다.

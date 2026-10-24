# 안전 모델

GPU 커널은 동시에 동일한 메모리를 사용하는 수천 개의 스레드를 실행합니다. CPU에서 Rust는 소유 및 빌림으로 데이터 경주를 방지합니다. 하나의 변동 가능한 참조, 별명 없이 컴파일 시에 강제됩니다. GPU에서는 SM 당 2048 개의 스레이드가 있습니다. 모두 동일한 기능에서 시작되며 모두 동일한 출력 버퍼를 가리키고 있습니다. 빌림 검사기는 이것에 설계되지 않았습니다.

cuda-oxide는 레이어로 문제를 해결합니다. 일반적인 경우 - 하나의 스레드는 체크된 발사 계약에 따라 하나의 요소를 작성합니다. 구조적으로 안전합니다. 비일반적인 경우 - 공유 메모리, 워프 셔플, 하드웨어 내장 함수 - 문서화 된 계약과 함께`unsafe`를 필요로합니다. 그리고 국경 사례 - TMA, 텐서 코어, 클러스터 수준의 통신 -는 완전히 수동적이며 제어하는 하드웨어의 복잡성에 부합합니다.

이 장은 모델을 설명하고 각 계층을 통해 걸어, 당신이 필요로 할 때 정확하게 알려줍니다`unsafe`그리고 왜.

---

## 3층

cuda-oxide는 컴파일러가 얼마나 확인할 수 있는지에 따라 커널 보안을 세 단계로 구성합니다:

|레벨       | 설명                                             | `unsafe`필요해요?|
|:-----------|:--------------------------------------------------------|:-------------------|
|1등급 | 안전 커널 본체와 체크된 `PreparedLaunch`        | 아니|
|** 2급 ** | 명확한 안전 계약과 함께 명시적인 `unsafe`           | 네, 스코프|
|** 3등급 ** | 원시 하드웨어 내장 함수 -- 사용자 책임     | 네, 퍼져나갑니다|

대부분의 응용 커널은 Tier 1 또는 Tier 1과 Tier 2를 넘어서며 전체적으로 살고 있습니다. Tier 3는 CUTLASS 또는 Triton IR 수준에서 구축하는 성능 엔지니어입니다. vecadd, GEMM 또는 감소를 작성하는 경우 Tier 2을 거의 떠나지 않습니다.

---

## 레벨 1: 기본 안전

### 핵심 아이디어: `DisjointSlice<T>` + `ThreadIndex`

기본 장치 측면 안전 추상화는 함께 레이스 무료 평행자를 지원하는 타입 쌍입니다.

- **`ThreadIndex<'kernel, IndexSpace>`** -  주위에 불투명한 목격자
`usize`공용 생성자가 없습니다. 하나를 얻는 유일한 방법은 신뢰할 수 있는 함수 (`index_1d`, `index_2d::<S>`) 는 ** 하드웨어 내장 변수 ** (`threadIdx`, `blockIdx`, `blockDim` 실행 시에 을 채우는 읽기 전용 특수 레지스터`ThreadIndex`입니다`!Send + !Sync + !Copy + !Clone`, 그리고 `'kernel`라이프타임은 커널 몸 안의 스택 로컬 스코프에 연결되므로 공유 메모리를 통해 스레드를 통해 밀수되거나 커널을 살아남을 수 없습니다.
- **`DisjointSlice<T, IndexSpace>`** -- 이 있는 슬라이스 같은 타입
`get_mut()`이 방법은 만 받아들입니다.`ThreadIndex`누구의 `IndexSpace`자신의 것과 일치합니다.`Option<&mut T>`- `None`범위를 벗어난 지수에서

그것들을 함께 넣으면 0의 커널을 얻을 수 있습니다`unsafe`

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

`get_mut_indexed`한 번의 호출 형태입니다: 그것은 스레드 단위 증인을 인쇄하고 로 해결합니다.`&mut T`한 장의 샷으로. 두 단계의 명시적인 형태는 여러 슬라이스에 평행 수학적 인덱스가 필요할 때도 사용할 수 있습니다:

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

안전은 다섯 가지 사실에서 나온다.

1. `index_1d()`는 X 좌표를만 사용합니다. 블록과이 있을 때 값이 고유합니다.
그리드 Y/Z 차원은 모두 1. A `domain = 1`준비된 발사 증명; 그 입력 된 증명 없이는 장치 Y/Z를 확인하고 증인이 유효하지 않다고 표시합니다.`None`.
2. `get_mut()`는 범위를 확인합니다. 범위 밖의 스레드는 `None`를 얻습니다.
3. `IndexSpace`매 매개 변수는 각 증인을 그 레이아웃에 묶습니다: a
`DisjointSlice<T, Index2D<128>>`는 `ThreadIndex<'_, Index2D<256>>`를 받아들이지 않습니다. - 스프레드를 혼합하는 것은 컴파일 오류입니다.
4. 증인인`!Send + !Sync + !Copy + !Clone`그리고 `'kernel`- 스코프,
그래서 스레드는 공유 메모리를 통해 서로 다른 인덱스를 세탁할 수 없습니다.
5. 체크된 `PreparedLaunch<K>`는 호스트 기하학을 정확한 커널과 연결합니다.

빌려주는 검사기는 스레드당 단일`&mut T` 를 볼 수 있다. 하드웨어는 좌표를 제공하며, 런칭 증명은 선형화 해체를 만든다. 타입 시스템은 장치와 호스트 증명을 결합시킨다.

```text
1D prepared launch: Y=Z=1 -> index_1d values are unique -> safe get_mut
raw 2D launch:       active Y/Z -> invalid witness -> safe get_mut returns None
```

원시 발사는 여전히 `unsafe`입니다: 호스트는 장치에서이 순위 검사가 종료되지 않더라도 다른 메모리 및 발사 의무를 유지해야합니다.

### 신뢰할 수 있는 인덱스 함수

`ThreadIndex`이 구조물은 구다-옥시드가 제공하는 구조물입니다:

|기능                      | 공식                                 | 반환 유형                                   | 참고문서|
|:------------------------------|:----------------------------------------|:----------------------------------------------|:-------------------------------------------------|
|`index_1d()`                  | `blockIdx.x * blockDim.x + threadIdx.x` | `ThreadIndex<'kernel, Index1D>`               | 비형식 발사시 Y/Z가 활성화되면 유효하지 않습니다|
|`index_2d::<S>()`             | `row * S + col`                         | `Option<ThreadIndex<'kernel, Index2D<S>>>`    | conststride; step를 혼합하는 것은 컴파일 오류입니다|
|`index_2d_runtime(&slice)`    | `row * width + col`                     | `Option<ThreadIndex<'kernel, Runtime2DIndex>>`| 호스트에 의해 묶여있는 슬라이스에서 읽은 줄 너비|
|`index_2d_row()`              | `blockIdx.y * blockDim.y + threadIdx.y` | `usize`                                       | 부품 액세서리, 목격자 제조가 아닌|
|`index_2d_col()`              | `blockIdx.x * blockDim.x + threadIdx.x` | `usize`                                       | 부품 액세서리, 목격자 제조가 아닌|

`index_2d_row()`그리고 `index_2d_col()``usize`-- 그들은 당신에게 수학의 구성 요소를 줄 것입니다. 하지만 로 인덱스를 만드는 데 사용할 수 없습니다.`DisjointSlice`선형화된 결과만 을 얻습니다.`ThreadIndex`마찬가지로 2차원 고유성은 발사에는 활성 Z 차원이 없다고 가정합니다.`domain = 2`준비된 발사도 증명합니다. 타입 지정 증명 없이, 장치는 Z를 확인하고 반환합니다.`None`가동되면

(일단 체크) =
### 타일을 한번 체크해

작은 고정 타일의 경우 한 바운드 체크가 그 타일의 모든 요소를 포함할 수 있습니다:

```rust
use cuda_device::{DisjointSlice, LinearTiles, kernel, launch_contract, thread};

#[kernel(launch_context = launch_context)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (64, 1, 1),
)]
pub fn add_one(mut values: DisjointSlice<u32, LinearTiles<4>>) {
    let thread_index = thread::index_1d_u32(launch_context);

    if let Some(mut tile) = values.tile_thread32(thread_index) {
        let first = tile.at_const::<0>().read();
        tile.at_const::<0>().write(first + 1);

        let second = tile.at_const::<1>().read();
        tile.at_const::<1>().write(second + 1);
    }
}
```

`LinearTiles<4>`는 스레드 0 요소 0-3, 스레드의 1 요소 4-7 등을 낸다. `tile_thread32`은 전체 4 요소 타일을 한 번 확인한다. 타일의 일부만 맞으면, `None`을 반환한다. `at_const::<0>()`와 같은 상수 위치들은 추가 실행 시간 경계 검사가 필요하지 않다. 유효하지 않은 상수는 컴파일에 실패한다.

두 가지 주석은 확인 된 사실을 제공합니다:

- `#[kernel(launch_context = launch_context)]`로컬 장치 값을 생성합니다
이 발사에 대한 사실을 가지고 있습니다. 그것은 커널 주장이 아닙니다.
- `domain = 1`는 X축만 활성화된다고 합니다.
- 이 경우, 이 `coordinates = u32`는 `thread::index_1d_u32(launch_context)`를 허용합니다.
이 스레드의 고유 인덱스를 전체 1차원 런칭에서 `u32`로 반환합니다.
- `block = (64, 1, 1)`각각의 블록에 정확히 64개의 스레드가 필요합니다.
아직 많은 블록이 있을 수 있습니다.

정확히`block`이 형태는 두 번, 독립적으로 실패하는 두 곳에서 확인됩니다. 준비는 모든 드라이버 호출 전에 호스트에서 일치하지 않는 시작 구성을 거부합니다. 형상은 장치 컴파일러에 도달하여 를 방출합니다.`.reqntid 64, 1, 1`, 그래서 CUDA 드라이버는 블록이 어떤 축에 다른 모든 발사 거부. 두 번째 체크는 첫 번째 볼 수 없습니다 경로를 커버:`unsafe`원시 발사, 그리고 충전된 PTX가 생성된  마커에 묘사된 유물 아닌 모듈.

`.reqntid`을 대체합니다`.maxntid`계약된 커널을 위한 것입니다. ptxas는 둘 다 선언하는 항목을 거부하고 정확한 형상이 이미 스레드 수를 제한합니다.`#[launch_bounds]`계속 `.maxntid`, 이 제품은 제한 `x * y * z`그리고 그 형상을 도달하는 모든 형상을 허용합니다. 그 차이점은 묶인 가닥이 블록 형상을 대신할 수 없는 이유입니다: 아래 `.maxntid 64, 1, 1`운전자가 동의합니다`(8, 8, 1)`그리고 `(4, 4, 4)`옆으로`(64, 1, 1)`.

2차원 타일용은 `thread::coord_2d_u32(launch_context)``RowMajorTiles<ROWS, COLS, ROW_STRIDE>`- `ROW_STRIDE`이 부분의 논리적 행 너비로, 버퍼의 레이아웃과 일치해야 합니다. 버퍼 길이는 행 너비가 곱할 필요는 없습니다. 전체 타일만 반환됩니다.

실행 가능한 `proof_carrying_views`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/proof_carrying_views) 예제에서 1차원 및 2차원 버전을 참조하십시오.

### 어떻게 `index_2d`가 타입 안전인가

`index_2d::<S>()`선발로 이동하는 상수변수에서 정수변수가 됩니다. 증인은 으로 돌아옵니다.`ThreadIndex<'kernel, Index2D<S>>`그리고 `DisjointSlice<T, Index2D<S>>`그냥 그 정확한 것을 받아들인다`S`두 개의 스레드는 서로 다른 걸음으로 증인을 만들어서 같은 슬라이스에 넣을 수 없습니다. 타입 시스템은 그것을 거부합니다.

```rust
#[kernel]
pub fn ok(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<128>() {  // matches
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}

#[kernel]
pub fn rejected(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<256>() { // ⛔ Index2D<256> != Index2D<128>
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}
```

증인은 또한 `!Send + !Sync + !Copy + !Clone`, 그리고 그것의 `'kernel` 수명은 매크로들이 주입하는 스택 로컬 스코프에서 빌려온 것이다. 그래서 스레드는 공유 메모리에 그것의 `ThreadIndex`을 주차할 수 없고 이웃이 나중에 그것을 가져오게 할 수 없으며, 증인은 커널 본체를 생존할 수 없다.

#### 정말 실행시간이 빨라지고 있어요:`index_2d_runtime`

일부 커널은 실제로 시작 시에 발걸음을 받습니다.`Runtime2DIndex`, 조각에서 발주 된 인덱스는:

```rust
let idx = thread::index_2d_runtime(&c)?;
```

아니`unsafe`, 왜냐하면 의견이 분분할 수 있는 호출 단위의 크기가 없기 때문입니다. 줄 너비는 슬라이스 안에 이동하고 호스트가 시작 패킷에 한 번 작성하므로 인덱스하는 모든 스레드는`c`같은 줄 너비로 구성합니다. 별도로 걸음을 통과하면 얻을 수 없습니다: 두 `ThreadIndex<'_, Runtime2DIndex>`다른 실행 시간 단계 아래에서 생성 값은 같은 유형을 가지고, 그래서 유형 시스템은 그들을 구별 할 수 없습니다. 당신은 컴파일 시간에 단계를 고정 할 수 있다면, 선호 `index_2d::<S>()`.

한 가지 결과는 증인 자체를 형성합니다. 두 가지`Runtime2DIndex`한 커널의 슬라이스는 다른 줄 너비를 가지고 있고, 그들의 증인은 타입을 공유합니다. 따라서 안전한 코드는 각 슬라이스에서 증인을 만들어서 어느 한쪽을 어느 슬라이스로 전달할 수 있습니다. 따라서 증인은 스레드의 원료를 저장합니다.`(row, col)`평면 지표보다는 좌표, 그리고 `get_mut`**주소** 슬라이스의 자체 행 너비에 대해 해결 (`row * width + col`, 필요`col < width`) 각 스레드는 서로 다른 좌표를 가지고 있으며 각 스레이드는 주소로 보내는 슬라이스로부터 동일한 호스트로 묶인 너비를 읽습니다. 따라서 매핑은 어떤 슬라이스가 증인을 발산했든 간에 슬라이스 당 주입이 계속됩니다. 발산 시간에 구운 평면 인덱스는 대신 발산 슬라이스의 격자를 밀수합니다: `(1, 0)`너비 5와 로 평평하게`(0, 5)`너비 100에 평평하게 두 이름 요소 5.

#### 실행시 줄 너비가 있는 곳

* 실행*는 "모든 스레드에서 동일한 값을 보았다"를 설정합니다: 호스트는 실행 패킷에 한 단어를 배치하고 모든 블록의 모든 스레드는 동일한 바이트를 읽습니다. 커널 몸에는 자체 로컬 중 하나에 대해 그것을 설정하는 비교 가능한 방법이 없습니다.

따라서, 잇몸이 실행시 행 너비를 가지고 있는 슬라이스는 같은 방식으로 그 너비를 받게 됩니다. 호스트에서, `RowWidth`는 이렇게 묶습니다:

```rust
module.write_c(&stream, &prepared, cuda_host::RowWidth::new(&mut c, n))?;
```

그리고 장치에서 액세서리는 자신의 논증 없이 다시 읽습니다:

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    // No `unsafe`: the row width came from the host, bound to `c`.
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

왜 이것은 충분합니까: 슬라이스당 한 줄 너비 아래에서 `last_col < stride` 체크는 각 타일을 한 논리적 줄 안에 유지하여 좌표에서 오프셋 매핑을 주입하게하므로 다른 스레드는 서로 다른 요소를 가지고 있습니다. 너비에 대해 동의하지 않는 두 스레드가 정확히 깨지는 경우입니다. 1x1 타일로, 스레드 5에서 스레드의`(1, 0)`는 인덱스 5을 평평하게 해소하고, 스레이드 100에서 스레이드는`(0, 5)` 또한 5: 하나의 요소, 두`&mut`를 해소합니다.

호출당 제공되는 너비는 각 후보 값이 자체적으로 실행 유니폼일 때에도 이를 배제할 수 없습니다. 보안 코드는 스레드 변동 조건 하에서 둘 중 하나를 선택할 수 있거나 한 슬라이스에서 두 호출 사이트에서 다른 하나를 전달할 수 있기 때문입니다. 호스트 경계에서 결합하면 선택이 제거됩니다.

묶인 값은 여전히 버퍼가 실제로 커널에 원하는 결과를 계산하는 줄 너비로 사용해야합니다. 이것은 안전이 아닌 정확성 요구 사항입니다. 레이아웃을 잘못 설명하는 너비는 슬라이스 내부에서 잘못된 답을 제공하며, 결코 위명 또는 범위를 벗어나는 액세스입니다.

#### 발사 일률 스칼라: `Uniform<T>`

같은 추론은 모든 스칼라에 적용됩니다. 커널은 줄 너비뿐만 아니라 모든 스레드에서 동일해야합니다. `Uniform<T>`이 사실은 타입으로 표시됩니다. 매개 변수를 선언하면 커널 ABI가 이 의무를 수행합니다:

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(rows: Uniform<u32>, mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    if coord.row() >= rows.get() {
        return;
    }
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

호스트 메소드는 여전히 평형 `u32`를 사용하며, `#[cuda_module]`는 그것을 랩하고, `#[repr(transparent)]`은 런칭 패킷 바이트를 동일하게 유지한다. 구성자가 없기 때문에 디바이스 코드는 스레드당 값을 증인으로 홍보할 수 없다. 다른 방법으로 검증된 값에 대해 `Uniform::new_unchecked`는 균일하게 남아 있으며, `block_dim_x()`와 그 형제들은 런칭이 하나의 블록 형상을 가지고 있기 때문에 직접 증인을 반환한다.

일률성은 모든 연산자가 일률적인 수학적 조건에서 폐쇄되어 있으므로 파생 값이 증인을 유지합니다. 선택 하에서 폐쇄되지 않습니다: 스레드 변동 조건에서 두 개의 일률적 값 사이에서 선택하면 스레드에 따라 다른 값을 제공합니다. 그렇기 때문에 이러한 종류의 증인은 전체 슬라이스를 통해 유지해야하는 증명을 수행 할 수 없습니다.

### GEMM 패턴

const-stride 2D 커널 (일반 사례 -- 타일 GEMM, 스텐실 커널, 고정 채널 수를 가진 이미지 커널) 에 대해서는, const-generic 형태가 안전 기본값입니다:

```rust
const STRIDE: usize = 1024;     // C is M x STRIDE

#[kernel]
pub fn gemm(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<STRIDE>>, m: u32) {
    let row = thread::index_2d_row();
    if let Some((c_elem, _)) = c.get_mut_indexed() {
        // col < STRIDE is guaranteed by `Some` -- no manual check needed
        if row < m as usize {
            // ... compute dot product into a local accumulator `sum` ...
            *c_elem = alpha * sum + beta * (*c_elem);
        }
    }
}
```

실행 시간 단계에 대해 동일한 형상이 작동하고 안전합니다. 행 너비는 일반 konst에서 아닌 출력 슬라이스로부터 나옵니다:

```rust
#[kernel]
pub fn gemm_runtime(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Runtime2DIndex>, m: u32, n: u32) {
    let row = thread::index_2d_row();

    // The row width comes from `c`, bound once by the host. No `unsafe`.
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        if row < m as usize {
            // ... compute dot product ...
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
```

`if let Some`에서`index_2d_runtime`설명서를 대체합니다`col < n`CUDA C++로 쓸 수 있는 방어를 `row < m`체크는 입력 매트릭스에서 쓰레기를 읽지 않도록 보호하기 때문에 유지됩니다.

출력자를 작성하는 것을 포함합니다. GEMM의 *입력* 쪽은, 모든 에 대한 한계 비용 체크입니다.`a[...]`읽은 -- 자신의 장을 가지고 있습니다: [한계 체크: 한 번 지불하거나 거부](bounds-checks.md).

### 1 차원 커널은 어떻게 만들어집니다?

커널과 발사도 완전히 안전합니다. 1단계입니다.

1. 모든 변동 가능한 출력 액세스 권한은 `DisjointSlice::get_mut(ThreadIndex)`
2. 모든 입력값은 공유된 변함없는 참조 (`&[T]`)
3. 공유 메모리, 무형 포인터, 스레드 인덱싱 이외의 본질적인 요소가 없습니다
4. 호스트는 일치 검사를 한 `PreparedLaunch<K>`

이 계층의 예는`vecadd`, `helper_fn`, `generic`, `host_closure`, 그리고 에 있는 순진한 GEMM 커널`gemm`그리고 `async_mlp`  이치 계약에 의해 실행됩니다.`LaunchConfig`경로는 안전하지 않습니다.

---

## 2등급: 범주 `unsafe`

모든 커널이 "하나의 스레드, 하나의 출력 요소" 패턴에 맞지 않습니다. 스레드가 협력해야 할 때 - 빠른 칩 내 메모리를 통해 데이터를 공유하거나 워크프로 레이너를 통해 통신하거나 원자 업데이트를 수행 할 때- 당신은 `unsafe`가 필요합니다. 2 차원의 주요 속성은 `unsafe`가 * 스코프 * 및 * 감사 * 하는 것입니다: 각 블록은 문서화된 안전 계약을 가지고 있으며, 나머지 커널은 안전합니다.

### 공유 메모리

공유 메모리는 빠르고 칩에 탑재되어 있으며 블록의 모든 스레드에 표시됩니다. 마지막 속성은 바로 이 때문에 `unsafe`가 필요합니다. - 대출 검사자는 동일한 `static mut` 배열에 작성되는 256 개의 스레드를 논할 수 없습니다.

```rust
static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

unsafe { TILE[ty * TILE_SIZE + tx] = value; }

thread::sync_threads();

let neighbor = unsafe { TILE[other_idx] };
```

계약: 동기화 없이 동시 스레드에서 충돌하는 글쓰기가 없도록 보장합니다. 이 작업이 이루어지는 도구는 `sync_threads()` 장애입니다. 모든 스레드가 스레드를 읽기 전에 글쓰기를 완료했다고 보장합니다

|API                       | 안전 의무|
|:--------------------------|:---------------------------------------------------------------------------------------------------|
|`SharedArray<T, N>`       | `static mut`를 통해 액세스합니다. 동기화 없이 충돌하는 글쓰기가 없습니다.|
|`DynamicSharedArray<T>`   | 동일한 규칙이지만 크기는 `LaunchConfig::shared_mem_bytes`를 통해 실행 시점에 설정됩니다.|

:::{seealso}
[공유 메모리와 동기화](../advanced/shared-memory-and-synchronization.md)전체 처리: 타일링, 은행 충돌, 동적 할당 및 이중 버퍼 파이프 라인
:::

### 워프 내적

워프 레벨의 원시자는 워프 내부의 스레드가 메모리에 전혀 영향을 주지 않고 데이터를 교환하도록 허용한다. 하드웨어에서 조정된 레지스터-레지스터 전송이다. 하드웨어가 스레드 컨버전스를 확인하지 않기 때문에 그들은 `unsafe`이다. 만약 당신이 분산 스레드를 포함하는 마스크를 통과한다면, 정의되지 않은 행동을 얻을 수 있다.

|API                                                                | 안전 의무|
|:-------------------------------------------------------------------|:-------------------------------------------------------------|
|`shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync`     | 소스 경로는 활성화되어야 합니다. 마스크는 호출 스레드를 포함해야 합니다.|
|`ballot_sync`, `any_sync`, #`all_sync`                              | 마스크의 모든 스레드는 으로 되어야 합니다.|
|`activemask`                                                       | 그 결과, 그 순간만 의미가 있습니다.|

:::{seealso}
[워프 레벨 프로그래밍]  내적을 사용하여 셔플 패턴, 감소 및 전사 합을 위한 ](../advanced/warp-level-programming.md)
:::

### 장애물 및 생명주기

`ManagedBarrier`typestate API는 장애물 수명 주기를 코드화합니다 (`Uninit`-> `Ready`-> `Invalidated`은 을 사용해서, 의 이`init()`그리고 `inval()`아직은 전환이 필요해요`unsafe`왜냐하면 그들은 하드웨어와 상호 작용하기 때문에, 하지만 타입 상태는 컴파일 시간에 가장 일반적인 실수를 방지합니다.

|API                                    | 안전 의무|
|:---------------------------------------|:--------------------------------------------------------------------------------|
|`mbarrier_init`                        | 정확히 하나의 스레드로 호출되어야 합니다. 바리어 공유 메모리에 있어야 합니다|
|`mbarrier_arrive` / `mbarrier_wait`    | 바리어가 초기화되어야 합니다. 토큰이 일치해야 합니다|
|`ManagedBarrier` (타입국)           | `init()`그리고 `inval()`요구합니다`unsafe`; 컴파일 시에 실행되는 상태 기계|

### 원자력

원자 연산은 유효한 원자 참조가 있으면 안전하게 호출할 수 있습니다.`unsafe`표면이 건설중입니다.`DeviceAtomicU32`원시 포인터에서 호출자가 포인터가 유효하고 올바르게 정렬된 것을 보장하도록 요구합니다:

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

// `from_ptr` is the unsafe constructor: it reinterprets existing memory.
// `DeviceAtomicU32::new(value)` is the safe one, for an owned static.
let atom = unsafe { DeviceAtomicU32::from_ptr(ptr) };
atom.fetch_add(1, AtomicOrdering::Relaxed);  // safe call
```

### 확인되지 않은 슬라이스 액세스

"하나의 스레드, 하나의 요소" 모델이 맞지 않을 때, 예를 들어, 단지 차선 0만이 결과를 기록하는 경전 수준 감소에서, `DisjointSlice::get_unchecked_mut(usize)`는 탈출 뚜을 제공합니다:

```rust
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    // SAFETY: Only lane 0 of each warp writes; warp indices are unique
    unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
}
```

보안 의무는 `ThreadIndex` 시스템이 자동으로 시행하는 것과 동일합니다: 경계 안에 인덱스, 두 개의 스레드가 동일한 인덱스를 공유하지 않습니다. 차이점은 타입 시스템이 당신을 위해 그것을하는 대신 직접 증명하는 것입니다.

이것은 접근에 대한 것입니다. 일반 뒤에 체크를 제거하기 위해`a[i]`한 번에 전체 커널을 인덱싱하는 것,  참조`unchecked_indexing`깃발 [한계 체크: 한 번 지불하거나 거부](bounds-checks.md).

---

## 레벨 3: 원료

스택의 하단에 있는 원시 하드웨어 내장 함수는 특정 GPU 아키텍처에 있는 특정 기능 단위와 직접 통신하는 API입니다. 모든 호출은 `unsafe`, 보안 계약은 복잡하고 아키테크처에 의존하며 문서는  문서 댓글보다 PTX ISA 매뉴얼에 더 많이 있습니다.

|특징                              | 주요 API                                                      | 건축물|
|:-------------------------------------|:--------------------------------------------------------------|:-----------------------|
|TMA (텐서 메모리 가속기)  | `tma_load_2d`, `tma_store_2d`, #`TmaDescriptor`                | sm_90+ (호퍼)|
|**tcgen05** (텐서 코어 제5)      | `tcgen05_mma`, `tcgen05_commit`, #`TensorMemoryHandle`         | sm_100a (블랙웰 DC)|
|**WGMMA** (전사단 MMA)            | `wgmma_mma_async`, `wgmma_commit_group`, #`wgmma_wait_group`   | sm_90+ (호퍼)|
|** 클러스터 **                          | `cluster_rank`, `map_shared_rank`, #`cluster_barrier_arrive`   | sm_90+ (호퍼)|
|CLC (클러스터 발사 제어)     | `clc_prefetch`, `clc_query_channel`                           | sm_100+ (블랙웰)|
|**TMEM** (텐서 메모리)             | `TmemGuard`(형태 상태), `tmem_alloc`, `tmem_dealloc`         | sm_100a (블랙웰 DC)|

응용 프로그램 수준의 커널을 작성하는 경우 Tier 3 API가 필요하지 않습니다. 다음 CUTLASS를 만드는 사람들을 위해 존재합니다. 그리고 그 사람들을 위해 cuda-oxide는 CUDA C++의 인라인 PTX와 동일한 하드웨어 액세스를 제공합니다.

:::{seealso}
[텐서 메모리 가속기] ](../advanced/tensor-memory-accelerator.md), [매트릭스 곱셈 가속자] ](../advanced/matrix-multiply-accelerators.md), 그리고 [클러스터 프로그래밍] [[%](../advanced/cluster-programming.md) 등 3단계 기능에 대한 자세한 설명을 위해.
:::

---

## 대출 체크가 주는 돈

cuda-oxide는 DSL 또는 매크로 시스템이 아닙니다. 그것은 커널 코드에서 실제 `rustc` 프론트엔드를 실행합니다. 즉 CPU에 rust가 제공하는 모든 안전 보장은 GPU에도 적용됩니다:

|보증                            | 어떻게 작동 합니까?|
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
|**소유 및 대출 **          | 라이프 타임 오류, 사용 후 무료, 그리고 위명 사용 위반 컴파일 시간에 잡힌|
|**보안 평행 기록**             | `DisjointSlice<T>`+ `ThreadIndex`+ 준비된 기하학과 일치하는 것은 글이 경주하지 않는다는 것을 증명합니다|
|**명료하게`unsafe`범위를 파악하는 방법**        | 원시 포인터 액세스에는 `unsafe`가 필요하며, 의무가 눈에 띄고 감사가 가능합니다.|
|** 컨버전트 속성 강제 | 시크 프림티브 (장벽, 울타리, 셔플)`convergent`IR에서 최적화 장치가 제어 흐름에 이동하거나 복제하는 것을 방지합니다|

첫 번째 세 가지는 표준 러스트입니다. 네 번째는 GPU 특이입니다: CUDA의 `bar.sync`, 울타리 및 워크 셔플 명령어는 컴파일러가 중복하거나 재질 수 없습니다. cuda-oxide는 IR에서 `convergent`를 표시하여 LLVM의 최적화 패스가 그들을 혼자 남겨줍니다.

---

## 어려운 문제들

러스트의 대출 검사기는 과 단일 스레드 소유를 위해 설계되었습니다.`Send`/`Sync`CPU 동시성. SIMT 실행은 대출 검사자가 결코 추론하는 법을 배우지 않은 패턴을 소개합니다. 여기 Cuda-oxide가 오늘날 적용하지 않는 것에 대한 정직한 회계입니다.

### 스레드 디버전트 제어 흐름

Rustc의 JumpThreading MIR 최적화는 if 명령어의 두 가지 지부 모두에 함수 호출을 복제합니다. CPU에서 완벽하게 작동하는 최적화이지만 블록의 모든 스레드가 같은 지점에서 을 어야 하는 GPU 장벽 의미론을 깨고 있습니다.`bar.sync`명령어. cuda-oxide는 현재 디바이스 코드 () 를 위해 JumpThreading를 비활성화합니다.`-Z mir-enable-passes=-JumpThreading`). 적절한 솔루션은 컴파일러에 컨버전스 요구 사항에 대해 가르쳐서 패스를 완전히 비활성화하기보다는 이를 최적화 할 수 있습니다.

### 공유 메모리 액세스 패턴

빌림 검사기는 스레드 0이 `smem[0]`를 쓰고 스레드가 1이 `smem[1]`을 쓰는 것이 안전하는지 여부를 판단할 수 없습니다. 그것은 `&mut smem` 을 보고 거부합니다. `DisjointSlice`는 고유 인덱스-쓰기 패턴을 해결하지만, 여러 스레드의 의도적으로 단계 간 동기화와 함께 중복되는 영역에 액세스하는 감소, 스캔 또는 생산자 / 소비자 파이프라인과 같은 협력 패턴은 해결하지 않습니다.

### 워프 레벨 컨버전스

`shfl_sync` 및 `ballot_sync`와 같은 연산은 참여 마스크에 이름을 붙인 모든 스레드가 실제로 호출 사이트에서 융합되어야 합니다. 타입 시스템은 오늘날 이것을 강제할 수 없습니다. 스레드들이 분리가 되어 전체 마스크를 통과하면, 침묵한 헝그를 얻습니다. 최악의 종류의 버그입니다. 왜냐하면 충돌도 오류 메시지도 없기 때문에, 결코 끝나는 커널만 있기 때문입니다.

### 메모리 공간 인식

GPU 메모리는 글로벌, 공유, 로컬, TMEM 등 다양한 주소 공간을 가지고 있다. 공유 메모리에 대한 `&mut`는 블록의 모든 스레드에 표시되며, 로칼 메모리에 있는 `&mut`는 하나의 스레드에만 비공개된다. 대출 검사기는 그것들을 동일하게 취급한다. 이것은 보수적이다 (어떤 안전한 프로그램을 거부하지만) 결코 불합리하지 않다 (불안한 프로그램을 받아들이지 않는다). 그럼에도 불구하고, 메모리 공간에 대한 인식이 있는 대출 검사기에는 `unsafe` 없이 더 많은 프로그램을 받아들일 수 있다.

### 왜 이 문제들이 해결될 수 있을까요?

의 타입 시스템에는 이미 빌딩 블록이 존재합니다. 확장되어야합니다.

|아이디어                                  | 해결 하는 방법|
|:--------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**집행자원 인식 유형**    | 실행 수준 (그리드 / 블록 / 워크 / 스레드) 으로 주석된 함수. 분산 브랜치 내부의 장애 호출은 컴파일 타임 오류가 된다.|
|**메모리 보기 **                      | 일반화된 병렬 액세스 패턴 - `DisjointSlice`처럼, 하지만 차단된, 줄무늬, 전이된, 그리고 구성된 레이아웃을 포함합니다. 스케일에서 타입 검증된 인종 자유.|
|** 연장된 대출 확인 동기화 ** | 장애물을 잊을 수 없거나, 분산 제어 흐름에 배치하거나, 최적화 장치에 의해 중복될 수 없도록 정적으로 강제합니다. 타입 시스템에서의 컨버전스.|

메모리 뷰 아이디어의 첫 번째 조각은 착륙했습니다: 실행 시간 크기의 행렬에 대한 행 및 열 읽기 뷰, 실행 시간 너비 출력 타일. [한계 체크: 한 번 지불 또는 선택] ](bounds-checks.md)는 그들을 다루고 있습니다.

이 모든 것은 컴파일 타임 분석입니다. 생성된 PTX는 손으로 작성하는 것과 동일합니다. 보안 네트워크는 코드 생성 시 사라집니다. 실행 시간 비용은 0입니다.

cuda-oxide는 이를 추가적으로 제공할 수 있는 좋은 위치에 있다. 실제 `rustc` 대출 검사기는 이미 디바이스 코드에서 실행된다. IR 인프라 (pliron 방언) 는 GPU 인식 분석 패스를 지원한다. MIR에서 PTX까지의 전체 컴파일 파이프 라인은 우리의 통제하에 있다. 그리고 각 새로운 안전 검사는 추가적이다. 기존 커널은 더 강력한 보증을 받으면서 컴파일을 계속한다.

---

## 안전한 커널을 작성하는 방법: 속임수

### 기본 경로

대부분의 커널에서 여기서 시작합니다:

```rust
#[kernel]
pub fn my_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
    if let Some((out, idx)) = output.get_mut_indexed() {
        *out = transform(input[idx.get()]);
    }
}
```

const-stride 2D를 위해, 슬라이스를 매개 변수화하고 const 인덱스를 요청합니다:

```rust
#[kernel]
pub fn tile_kernel(mut output: DisjointSlice<f32, Index2D<1024>>) {
    if let Some((out, _idx)) = output.get_mut_indexed() {
        *out = ...;
    }
}
```

규칙:

- 모든 변수 출력값에 `DisjointSlice`를 사용한다.
- 모든 읽기 전용 입력에 `&[T]`를 사용한다.
- 1D 그리드에서는 `launch_contract(domain = 1, ...)`를 선언하고 준비된을 사용하십시오
시작. 커널 내부에서, 기본으로 `get_mut_indexed()`로 설정합니다. 여러 슬라이스에 대한 수학적 인덱스가 필요하다면, 명시적인 쌍으로 돌아가십시오: `let idx = thread::index_1d(); slice.get_mut(idx)`.
- const-stride 2D 그리드에서는, 로 슬라이스를 매개변수화합니다.
`DisjointSlice<T, Index2D<S>>`그리고 사용`get_mut_indexed()`또는`thread::index_2d::<S>()`. 맞지 않는 걸음걸이는 컴파일 오류입니다.
- 실행 시간 단계에 대해, 으로 `thread::index_2d_runtime(&slice)`을 도달
이 슬라이스는 `Runtime2DIndex`-태그된 슬라이스입니다. 줄 너비는 슬라이스의 내부로 이동하고 호스트에 의해 한 번 묶여 있으므로 동의하지 않는 `unsafe` 및 통화 단위의 단계가 없습니다.
- 항상 `get_mut()` / `get_mut_indexed()`를 통해 경계 확인 (둘 다 반환)
`Option`).

커널 본체가 `unsafe` 없이 컴파일되면, 디바이스 사이드 부분을 만족시킨다. 레이스 자유는 또한 매칭 준비된 런칭, 또는  구성에 대한 명시적인 unsafe 증명이 필요하다.

### 필요할 때`unsafe`

|패턴              | 왜?                                           | 완화|
|:---------------------|:----------------------------------------------|:--------------------------------------------------------------------|
|공유 메모리        | 여러 개의 스레드가 동일한 `static mut`에 액세스합니다 | 크로스 스레드 읽기 전에 `sync_threads()`와 동기화|
|워프 셔플        | 스레드 컨버전스가 컴파일러에서 확인되지 않습니다    | 풀 워프 동작에 `FULL_MASK`를 사용한다. 부분 마스크를 문서화한다.|
|원자력              | 원시 포인터에서 만든 것               | 원자력 작전 자체는 안전합니다|
|비일체적인 글쓰기   | 모든 스레드가 자신의 인덱스에 글을 쓰는 것은 아닙니다.      | 문서화된 고유성 논증과 함께 `get_unchecked_mut`를 사용하십시오|
|하드웨어의 내장 함수  | 복잡하고 건축에 특화된 계약      | PTX ISA 문서를 따르고 목표 하드웨어에 테스트|

### `SAFETY` 댓글

모든 `unsafe`블록, 문서 *왜* 변수가 유효합니다. 코드가 하는 것이 아니라 - 코드가 이미 그렇게 말합니다 - 하지만 왜이 특정 사용이 안전합니다:

```rust
// SAFETY: Only lane 0 of each warp executes this branch.
// Warp indices (gid / 32) are unique across warps, so no two
// threads write to the same output element.
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    unsafe { *partial_sums.get_unchecked_mut(warp_idx) = warp_sum; }
}
```

이것은 의식이 아닙니다. 커널이 새벽 2시에 데이터 레이스를 하고`compute-sanitizer`로그, 너한테서 안전성 댓글을 받은 게 버그에 가장 빠른 길이야

:::{tip}
설득력 있는 글을 쓸 수 없다면`SAFETY`에 대한 댓글`unsafe`블록, 즉 변수가 실제로 유지되지 않는다는 신호입니다. 문자가 명백할 때까지 코드를 재구성하거나 안전한 API를 사용하십시오.
:::

---

## 요약

|부동산                                                        | 상태|
|:----------------------------------------------------------------|:-----------------------------------------------------|
|디바이스 코드에서 빌려보게                                   | 강제 (실제 `rustc` 프론트엔드)|
|안전 1D 병렬 기록 (`DisjointSlice + index_1d`)            | 으로 `domain = 1`준비된 발사|
|안전 2D 평행은 -- const stride                         | 강제 (`Index2D<S>` 미합격은 컴파일 오류)|
|안전 2D 평행 표시 -- 실행 시간 스프레드                       | 강제 (주체로 연결된 행 너비, 조각당 해상도)|
|`ThreadIndex` 스레드 간 전송이 불가능 (스메임 세탁) | 강제 (`!Send + !Sync + !Copy + !Clone + 'kernel`)|
|`&mut [T]` 커널 매개 변수                                     | 강제되지 않습니다 -- 모든 `&mut`의 논제를 `unsafe`로 취급합니다|
|공유 메모리, 내장 함수성`unsafe`를 위한 명시적 `unsafe`0$]]                 | 강제 (강성 언어 규칙)|
|동시 원시자들에 대한 컨버전트 속성                         | 강제 (IR 레벨의 `convergent` 표시)|
|워프 작전용 스레드 컨버전스                                 | 강제하지 않음 (운전 기간 의무)|
|메모리 공간 인식 (공유 대 글로벌)                       | 강제하지 않습니다 (미래 작업)|

`&mut [T]`커널 매개 변수로 다음의 미공점입니다: 매크로는 유형을 받아들이지만 실행 시간 레이아웃 (모든 스레드는 동일한 지원 포인터를 볼 수 있습니다) 는 을 통해 작성하는 것을 의미합니다.`&mut data[i]`두 개의 다른 스레드에서 같은 종류의 위명인 `DisjointSlice`막론하고 거부할 때까지`&mut [T]`정답으로 (또는 다시 쓰면 `DisjointSlice`), 1을 사용하는 모든 커널을 그 안에 있는 모든 줄이 인 것처럼 취급합니다.`unsafe`.

안전 모델은 커널 보디와 준비된 실행를 통해 일반적인 경우를 안전하게 만들 수 있도록 설계되었으며, 다른 모든 것에 대해 명시적인 탈출 래치를 제공합니다. 컴파일러와 실행 계약이 운반하지 않는다는 증거를 제공 할 때만 `unsafe`를 사용하십시오.

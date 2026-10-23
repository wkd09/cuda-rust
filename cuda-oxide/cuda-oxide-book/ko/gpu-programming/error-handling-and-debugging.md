# 오류 처리와 디버깅

GPU 커널은 CPU 코드와 다른 방식으로 실패합니다. 현재 CUDA 툴체인은 예외나 스택 언와인드를 지원하지 않고 커널 출력에는 스택 추적이나 `println!`이 없습니다. 문제가 생기면 조용한 데이터 손상, 하드웨어 트랩, 호스트에서 발생하는 난해한 드라이버 오류로 나타납니다. 이 장에서는 cuda-oxide의 커널 문제 진단 및 해결 도구를 설명합니다.

## 커널 오류가 발생하면

GPU 오류는 네 가지 유형으로 나뉩니다.

| 실패 유형          | 관찰되는 현상                                  | 예시                                                |
|:----------------------|:----------------------------------------------|:-------------------------------------------------------|
| **조용한 손상** | 오류 없이 잘못된 결과                       | 데이터 경합, 인덱스 off-by-one                       |
| **하드웨어 트랩**     | 동기화 시 `CUDA_ERROR_ILLEGAL_INSTRUCTION`      | `debug::trap()`, 메시지 없는 `gpu_assert!`, 패닉, 범위 초과  |
| **디바이스 어서션**  | 어서션 진단과 `CUDA_ERROR_ASSERT` | `gpu_assert!(condition, "message")`                    |
| **실행 실패**    | `DriverError` 즉시 반환            | 그리드 차원 오류, 모듈 누락, 리소스 부족      |

현재 CUDA 툴체인은 예외나 스택 언와인드 메커니즘을 제공하지 않습니다. 하드웨어 트랩은 구조화된 어서션 메타데이터 없이 커널을 종료합니다. CUDA의 디바이스 측 `__assertfail` 시스템 호출도 실행을 종료하지만, 동기화 시 `CUDA_ERROR_ASSERT`를 반환하기 전에 어서션 메시지와 호출 위치 메타데이터를 보고합니다.

두 오류 모두 비동기 디바이스 오류이며 보통 스트림이나 컨텍스트를 동기화할 때 나타납니다. 둘 다 CUDA 컨텍스트를 사용할 수 없게 만들지만 복구 방법은 다릅니다. 디바이스 어서션(`CUDA_ERROR_ASSERT`) 뒤에는 컨텍스트를 삭제하고 다시 만들어 CUDA를 계속 사용할 수 있습니다. 하드웨어 트랩(`CUDA_ERROR_ILLEGAL_INSTRUCTION`) 뒤에는 CUDA가 프로세스 전체를 손상된 상태로 간주하므로 프로세스를 종료하고 다시 실행해야 합니다.

## `gpu_printf!` — GPU에서 출력

`gpu_printf!`는 빠른 디버깅을 위해 디바이스 코드에서 값을 출력합니다. CUDA 내장 `vprintf` 메커니즘을 사용합니다.

```rust
use cuda_device::{DisjointSlice, gpu_printf, kernel, thread};

#[kernel]
pub fn debug_kernel(data: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if idx.get() < 4 {
        gpu_printf!("Thread {} sees value {}\n", idx.get(), data[idx.get()]);
    }
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = data[idx.get()] * 2.0;
    }
}
```

타입 문자가 없는 플레이스홀더는 인자 타입에 따라 출력합니다. 위 예제의 `usize` 인덱스는 `%llu`, `f32` 원소는 `%f`로 출력됩니다. `{:x}` 또는 `{:.2f}`는 인자를 해당 포맷이 지정하는 C 타입으로 변환합니다.

### 주요 사항

- **출력을 보려면 동기화해야 합니다.** 출력은 GPU에 버퍼링되며 스트림 또는 디바이스 동기화(`to_host_vec`, `ctx.synchronize()` 등) 뒤에 호스트에 나타납니다.
- **버퍼 크기.** 기본 printf 버퍼는 1MiB입니다. 많은 스레드가 출력하면 내용이 잘릴 수 있습니다. `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size)`로 늘리세요.
- **스레드 순서.** 서로 다른 스레드의 출력 순서는 임의적입니다.
- **성능.** printf는 스레드 간 실행을 직렬화하므로 핫패스에서 피하세요. 로깅이 아니라 디버깅에 사용하세요.
- **포맷 변환.** 매크로는 컴파일 시 Rust `{}` 포맷 지정자를 C printf 형식(`%d`, `%f` 등)으로 변환합니다.

### `println!`이나 `Debug`를 사용할 수 없는 이유

표준 Rust 포매팅(`fmt::Display`, `fmt::Debug`, `format!`, `println!`)에는 동적 디스패치, 문자열 할당, 입출력이 필요하지만 GPU에는 이런 기능이 없습니다. `gpu_printf!`는 CUDA `vprintf` 호출로 직접 하위 변환해 이를 우회합니다.

## `gpu_assert!`, `__assertfail`, `trap()`

디바이스에서 치명적인 오류를 검사하려면 `gpu_assert!` 또는 `debug::trap()`을 사용하세요.

```rust
use cuda_device::{DisjointSlice, debug, gpu_assert, kernel, thread};

#[kernel]
pub fn checked_kernel(data: &[f32], len: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    // No-message form: lowers to trap()
    gpu_assert!(idx.get() < len as usize);

    // String-literal message form: lowers to CUDA's __assertfail
    gpu_assert!(
        data[idx.get()] >= 0.0,
        "expected non-negative input"
    );

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = data[idx.get()];
    }
}
```

| 연산                           | 디바이스 동작                         | 호스트 결과                          |
| :---------------------------------- | :-------------------------------------- | :----------------------------------- |
| `gpu_assert!(condition)`            | 거짓이면 `trap()` 실행              | `CUDA_ERROR_ILLEGAL_INSTRUCTION`     |
| `gpu_assert!(condition, "message")` | CUDA 디바이스 측 `__assertfail` 호출 | 진단 메시지와 `CUDA_ERROR_ASSERT`  |
| `debug::trap()`                     | 무조건 트랩 실행          | `CUDA_ERROR_ILLEGAL_INSTRUCTION`     |
| `debug::breakpoint()`               | `brkpt` 출력                           | cuda-gdb에서 일시 중지, 디버거가 없으면 실패 |

`gpu_assert!`에 전달하는 메시지는 문자열 리터럴이어야 합니다. 포맷된 어서션 메시지는 현재 지원되지 않습니다.

### 실행 후 동기화 패턴

디바이스 오류는 보통 스트림 동기화 시 보고됩니다.

```rust
// Launch kernel.
// SAFETY: config matches vecadd's 1D indexing and all buffer bounds.
unsafe { module.vecadd(&stream, config, &a, &b, &mut c) }
    .expect("Launch failed");

// Surface asynchronous traps or device assertions.
stream.synchronize().expect("Kernel failed on the device");
```

`gpu_assert!(condition, "message")`를 사용하면 CUDA 드라이버가 동기화 시 `CUDA_ERROR_ASSERT`를 반환하기 전에 메시지, 소스 파일, 줄 번호, 모듈 정보를 출력합니다.

메시지가 없는 형식은 어서션 메타데이터를 포함하지 않습니다. 실패한 검사를 식별해야 한다면 메시지 형식을 쓰고, 런타임 값도 확인해야 한다면 `gpu_printf!`를 사용하세요.

일반 Rust의 `assert!`, `debug_assert!`, `panic!` 경로는 변경되지 않으며 기존 트랩 기반 동작을 계속 사용합니다.

## 호스트 측 오류 처리

### `DriverError`

동기 실행 경로는 `Result<(), DriverError>`를 반환합니다. `DriverError`는 CUDA 드라이버 결과 코드를 감쌉니다.

```rust
// SAFETY: config matches vecadd's 1D indexing and all buffer bounds.
match unsafe { module.vecadd(&stream, config, &a, &b, &mut c) } {
    Ok(()) => { /* launched successfully */ }
    Err(e) => eprintln!("Launch failed: {e}"),
}
```

### `DeviceError`

비동기 경로(`{kernel}_async` / `DeviceOperation`)는 드라이버 오류와 컨텍스트 및 스케줄링 오류를 함께 감싸는 `DeviceError`를 사용합니다.

```rust
use cuda_async::simt::error::DeviceError;

let result: Result<Vec<f32>, DeviceError> = operation.sync();
```

`DeviceError` 변형에는 `Driver`, `Context`, `KernelCache`, `Scheduling`, `Launch`, `Internal`이 있습니다.

### `CudaContext::check_err`

여러 작업을 실행한 뒤 기록된 비동기 오류를 확인하려면 컨텍스트에서 `check_err()`를 호출하세요.

```rust
ctx.check_err().expect("Asynchronous GPU error detected");
```

## `cargo oxide debug` — cuda-gdb 통합

`cargo oxide debug`는 디버그 정보를 포함해 커널을 빌드하고 cuda-gdb를 실행합니다.

```bash
cargo oxide debug vecadd          # Standard GDB
cargo oxide debug vecadd --tui    # GDB with TUI
cargo oxide debug vecadd --cgdb   # cgdb front-end
```

기본 설정은 **소스 수준 디버깅**입니다. cuda-gdb가 Rust 소스 파일에서 멈추고 유용한 백트레이스를 표시합니다. 지역 변수 검사는 별도의 더 무거운 모드이며 필요할 때 선택할 수 있습니다.

### 디버그 정보 모드

cuda-oxide에는 디바이스 디버그 모드가 세 가지 있습니다.

| 모드 | 활성화 방법 | 제공 기능 | 비용 |
|:-----|:-----------------|:-------------|:-----|
| 끔 | 일반 `build` / `run`의 기본값 | 가장 빠른 PTX, 소스 매핑 없음 | 없음 |
| 라인 테이블 | `--lineinfo`, `cargo oxide debug`, `CUDA_OXIDE_DEBUG=line-tables` | 소스 중단점, 단계별 실행, 백트레이스 | 낮음 |
| 전체 | `--device-debug`, `CUDA_OXIDE_DEBUG=full` | 라인 테이블과 기본 인자/지역 변수 검사 | 높음 |

라인 테이블은 기계어 명령을 소스 줄로 연결하는 지도와 같습니다.

```text
PTX instruction  ──debug line table──>  src/main.rs:39
```

Full debug adds variable records:

```text
source local `tid`
      |
      v
LLVM/DWARF says: "tid lives in this stack slot"
      |
      v
cuda-gdb can try: print tid
```

지역 변수를 확인하려면 현재 명령이 해당 변수와 같은 어휘적 범위 안에 있어야 합니다.

```text
function
  └─ if-let block
      └─ loop block
          └─ current instruction
```

커널 실행 지점이나 첫 헬퍼 호출처럼 너무 일찍 멈추면 변수가 아직 레지스터에 로드되지 않아 `<optimized out>`으로 표시될 수 있습니다. 변수 확인 시에는 값이 사용된 뒤의 소스 줄에 중단점을 두세요.

```text
break src/main.rs:412
run
info args
info locals
```

변수가 `<optimized out>`으로 표시된다고 해서 반드시 컴파일러 버그는 아닙니다. 그 시점의 PC에서 해당 값이 살아 있는 기계어 위치를 갖지 않는다는 뜻일 수 있습니다. 디버그 정보는 지도이지 타임머신이 아닙니다.

인라인된 헬퍼 호출에서 cuda-oxide는 각 인자의 원래 소유 함수도 보존합니다. 서로 다른 두 함수에서 모두 인자 번호 `1`을 사용할 수 있으므로 중요합니다.

```text
kernel(data) calls helper(self)

data -> arg #1 in kernel's debug scope
self -> arg #1 in helper's debug scope, with "inlined at" the kernel callsite
```

범위가 분리되지 않으면 LLVM은 메타데이터가 서로 모순된다고 판단해 버립니다. 디버그 정보에는 성만이 아니라 계보까지 필요합니다.

먼저 라인 테이블을 사용하세요. 대부분의 “실행이 어디로 갔나?” 질문에 충분하며 느린 CUDA 디버그 타깃 모드를 피할 수 있습니다. `print idx`, `print ptr`처럼 지역 변수를 검사해야 할 때 전체 디버그를 사용하세요. 디버거는 많은 정보를 볼 수 있지만 항상 빠르지는 않습니다.

`CUDA_OXIDE_DEBUG` 재정의는 `build`, `run`, `pipeline`, `debug`에서 사용할 수 있습니다.

```bash
CUDA_OXIDE_DEBUG=line-tables cargo oxide pipeline vecadd
CUDA_OXIDE_DEBUG=full cargo oxide debug vecadd
```

각 모드에는 대응하는 플래그도 있으며 환경 변수는 그 축약형입니다. `--lineinfo`는 라인 테이블, `--device-debug`는 전체 디버그를 선택하며 `nvcc -lineinfo`, `nvcc -G`와 대응합니다. 두 플래그는 `build`, `run`, `pipeline`, `test`, `sanitize`, `inspect`, `emit-ltoir`에서 사용할 수 있습니다.

```bash
cargo oxide build vecadd --lineinfo
cargo oxide run vecadd --device-debug        # same as CUDA_OXIDE_DEBUG=full
```

두 규칙으로 함께 지정했을 때 동작이 결정됩니다. `--device-debug`가 `--lineinfo`보다 우선하므로 둘 다 전달하면 전체 디버그가 됩니다. 두 플래그를 *생략*해도 디버그가 꺼지는 것은 아닙니다. 플래그가 설정을 요청할 때만 빌드에 `CUDA_OXIDE_DEBUG`를 내보내므로, 플래그가 없으면 외부 환경이 지정한 수준을 그대로 유지합니다. 플래그가 있으면 해당 값을 내보내 상속된 값보다 우선합니다.

Useful aliases:

| 값 | 의미 |
|:------|:--------|
| `off`, `none`, `0` | 디바이스 디버그 메타데이터 없음 |
| `line-tables`, `line`, `lines`, `1` | 소스 라인 테이블만 |
| `full`, `2` | 라인 테이블과 기본 변수 메타데이터 |

### 전체 디버그가 최적화에 미치는 영향

전체 디버그는 지원되는 지역 변수를 cuda-gdb가 읽을 수 있는 안정적인 메모리 위치에 유지합니다.

```text
rustc MIR                         cuda-oxide / LLVM
------------------------------    ------------------
ScalarReplacement...: off    ->   mem2reg: off
SingleUseConsts: off               loop unrolling: off
ReferencePropagation: on           opt -O2: off
general inlining: on*              llc: -O0

* only DisjointSlice::get_mut is outlined
```

`ReferencePropagation`은 포인터 대입을 `AssignRef` 디버그 이벤트로 바꿀 수 있습니다. 전체 모드는 지원되는 슬라이스 인덱스 형식을 다음과 같이 처리합니다.

```text
ptr = &slice[index] -> AssignRef -> debugger stack slot -> cuda-gdb
```

고정 크기 배열을 직접 가리키는 불변 참조도 같은 제한된 스택 홈 연결 방식을 사용합니다.

```text
&array[index]
    -> AssignRef
    -> debugger stack slot
    -> cuda-gdb
```

다중 값 `DIArgList` 형식은 LLVM IR에서 주소 계산식을 표현할 수 있지만, 결과 포인터 위치가 런타임 인덱스에 따라 달라 `ptxas` 이후에 안정적으로 검사할 수 없으므로 이 생성기에서는 사용하지 않습니다. 따라서 초기 배열 확장은 고정 크기 배열에 대한 직접 런타임 `usize` 인덱스 하나만 허용하며, 재구성된 참조는 전체 디버그 모드에서만 실체화합니다. 주변의 필드/역참조 체인은 계속 지원하지 않습니다.

지원되지 않는 이벤트는 부분 값을 출력하는 대신 해당 디버그 바인딩을 생략합니다. 스택 홈 저장은 전체 디버그 빌드에서만 존재합니다. 임포터가 아직 변환하지 못하는 MIR 형식이 나타나므로 `-Zmir-opt-level=0`은 사용하지 않습니다. rustc가 대응하는 디버그 이벤트 없이 제거한 지역 변수는 파이프라인 뒤쪽에서 복구할 수 없습니다.

`CUDA_OXIDE_NO_OPT=1`을 설정할 필요는 없습니다. 전체 모드가 위에 표시한 후속 최적화 단계를 이미 비활성화합니다.

| 설정 | 의미 |
| :------ | :------ |
| `CUDA_OXIDE_DEBUG=off` | 디바이스 디버그 메타데이터 없음. 완전히 최적화된 PTX |
| `CUDA_OXIDE_DEBUG=line-tables` | 소스 줄 정보만. 최적화는 유지 |
| `CUDA_OXIDE_DEBUG=full` | 소스 줄 및 지원 지역 변수/인자, 선택적 MIR 제어, 후속 최적화 비활성화 |

라인 테이블은 최적화 후에도 줄 매핑이 잘 유지되므로 최적화 파이프라인에서 사용됩니다. 전체 모드는 선택적 MIR 호환 정책을 유지하면서 지원 지역 변수가 실제로 남도록 cuda-oxide의 후속 최적화 단계를 끕니다.

> Pliron `mem2reg`가 승격을 고려해 수행하는 `mir.dbg_value` 복구는 향후 *최적화된* 디버그 단계(가능한 범위에서 `opt -O2`를 거친 지역 변수)의 기반입니다. 현재 `full` 모드는 이를 사용하지 않습니다.

### 현재 지원 범위

라인 테이블 모드는 다음을 지원합니다.

- 커널 이름을 이용한 중단점 설정(예: `break vecadd`)
- 소스 단계별 실행과 백트레이스
- 다른 파일의 헬퍼/인라인 소스 위치 확인(예: 커널에서 `cuda-device/src/thread.rs`로 단계별 이동)

전체 모드(`-G`)에서는 다음을 검사할 수 있습니다.

- rustc가 `var_debug_info`로 노출하는 지역 변수와 인자
- 스칼라 타입(`bool`, 정수, 부동소수점), 원시 포인터, 참조
- 구조체, 튜플, 고정 크기 배열의 필드(실제 레이아웃에 맞는 오프셋으로 표시), 예:
  `out = DisjointSlice {ptr: 0x..., len: 1}` and `idx = ThreadIndex {raw: 0}`

디바이스 정적 변수는 cuda-gdb를 Rust 모드로 두고 크레이트 경로를 포함한 이름으로 접근하세요.

```text
set language auto
print device_global::DEVICE_COUNTER
print constant_memory_simple::kernels::SCALE
```

```text
device_global::DEVICE_COUNTER          -> @global u64
constant_memory_simple::kernels::SCALE -> ConstantMemory<f32>, AS4
```

상수 메모리 검증기는 cubin에서 `DW_AT_address_class 4`도 확인합니다.

중단점 바인딩, 백트레이스, `info args`/`info locals` 등 전체 동작은 실제 하드웨어에서 `cuda-oxide/scripts/debug-smoketest.sh`로 확인합니다.

지원 범위는 아직 제한적입니다. ABI 경계에서 `(ptr, len)` 쌍으로 분리된 슬라이스 인자, 반복 역참조, 부분 슬라이스, 여러 런타임 인덱스, 필드/역참조 체인이 포함된 런타임 인덱스 투영, 역참조+인덱스/다운캐스트 체인, 필드가 아닌 복합 조각은 지원하지 않습니다. 정적 구조체/튜플 필드, 고정 배열 상수 인덱스, 열거형 페이로드 다운캐스트, 얇은 포인터/참조 1회 역참조, 고정 배열을 가리키는 문장 디버그 참조의 직접 런타임 `usize` 인덱스 1개는 지원됩니다. 다른 인라인 헬퍼 프레임의 지역 변수는 커널 지역 변수보다 정보가 적을 수 있으므로 소스 바인딩을 소유한 프레임을 선택하세요.

### 중단점 작업 흐름

1. 디버그 빌드: `cargo oxide debug <example>`
2. 커널에 중단점 설정: `break vecadd`
3. 실행: `run`
4. 스레드 검사: `cuda thread`, `cuda block`, `cuda warp`
5. 변수 출력: `print idx`, `print *c_elem`

`cargo oxide debug`는 `target/release/<example>`을 가정하지 않고 Cargo 빌드 아티팩트 메타데이터에서 실행 파일을 선택합니다. 따라서 사용자 지정 바이너리 이름, `package.default-run`, 설정된 타깃 디렉터리, 호스트 타깃 트리플, 가상 워크스페이스에서도 디버깅할 수 있습니다. 패키지에 바이너리가 여러 개 있고 기본값이 명확하지 않으면 `--bin <name>`을 사용하세요.

코드에서 중단점을 설정하려면 커널에 `debug::breakpoint()`를 사용하세요. cuda-gdb가 `brkpt` 명령에 도달하면 실행을 일시 중지하고 GPU 상태를 검사할 수 있습니다.

:::{tip}
디버거가 연결되지 않은 상태에서 `debug::breakpoint()`를 실행하면 커널이 **충돌합니다**. 컴파일 시 플래그로 보호하거나 디버깅 세션에서만 사용하세요.
:::

## `cargo oxide sanitize` — Compute Sanitizer

메모리와 동기화의 정확성을 검사하려면 NVIDIA Compute Sanitizer에서 같은 빌드 경로를 실행하세요.

```bash
cargo oxide sanitize vecadd
cargo oxide sanitize sharedmem --tool racecheck
cargo oxide sanitize debug --tool synccheck -- --kernel-name kns=clock
cargo oxide sanitize my_app -- --leak-check full -- --app-flag value
```

기본 도구는 `memcheck`입니다. 공유 메모리 위험에는 `racecheck`, 초기화되지 않은 전역 메모리 읽기에는 `initcheck`, 잘못된 동기화에는 `synccheck`를 사용하세요. `--` 뒤의 추가 인자는 실행 파일 앞에서 `compute-sanitizer`에 직접 전달됩니다. 대상 프로그램에 인자를 전달하려면 실행 파일 뒤에 두 번째 `--`를 사용하세요.

진단 결과에 Rust 소스 파일과 줄 번호를 표시하도록 이 명령은 기본적으로 최적화된 디바이스 라인 테이블을 켭니다. 프로세스나 프로젝트에서 명시한 `CUDA_OXIDE_DEBUG` 설정은 계속 우선합니다.

Compute Sanitizer는 도구 진단을 보고해도 보통 종료 코드 0을 반환합니다. 따라서 `cargo oxide sanitize`는 기본값으로 `--error-exitcode 86`을 전달해 진단이 스크립트와 CI에서 실패로 처리되게 합니다. 명시적으로 전달한 sanitizer 인자는 이 기본값을 재정의합니다.

```bash
# Intentionally keep a zero exit status and inspect the printed report.
cargo oxide sanitize vecadd -- --error-exitcode 0
```

`--check-exit-code no`, `--require-cuda-init no` 같은 옵션은 종료 코드 0이 의미하는 바를 약하게 만듭니다. 따라서 래퍼는 프로세스 상태만으로 보고서가 깨끗하다고 판단하지 않습니다. 완료 상태를 표시하고 sanitizer 출력을 확인할 수 있도록 남겨 둡니다.

CLI 플래그 `--no-fmad`는 일반 빌드와 상호 운용 빌드 모두에 전달됩니다. 일반 곱셈과 덧셈/뺄셈을 분리하고 각 연산마다 한 번 반올림합니다. `f32::mul_add` 같은 명시적 융합 연산은 계속 융합됩니다.

NVVM IR 및 LTOIR 빌드는 `.options`와 버전이 지정된 `.target` 사이드카도 생성합니다. 이후 libNVVM과 nvJitLink 단계에서 같은 FMA 정책을 유지하도록 아티팩트와 함께 두 파일을 보관하세요.

메모리 안전성을 조사할 때는 먼저 `memcheck`를 실행하세요. 다른 도구는 보완적이며 전체 메모리 접근 검사를 수행하지 않습니다.

## `cargo oxide doctor` — 환경 검사

커널 오류를 디버깅하기 전에 환경이 올바르게 구성되었는지 확인하세요.

```bash
cargo oxide doctor
```

doctor가 검사하는 항목은 다음과 같습니다.

| 검사 항목           | 확인 내용                               |
|:----------------|:-----------------------------------------------|
| Rust 툴체인  | 필수 구성 요소가 설치된 nightly 컴파일러      |
| 코드 생성 백엔드 | `librustc_codegen_cuda.so` 빌드 여부               |
| CUDA 헤더    | 툴킷 루트에 `cuda.h`가 있는지        |
| CUDA 툴킷    | `nvcc` 검색 및 버전 호환 여부            |
| libNVVM         | `libnvvm.so` 로드 가능 여부(libdevice 수학 커널) |
| nvJitLink       | `libnvJitLink.so` 로드 가능 여부(동일)              |
| libdevice       | `libdevice.10.bc` 검색 가능 여부(동일)          |
| LLVM            | PTX 생성을 위한 `llc`(21+) 사용 가능 여부       |
| 드라이버/GPU    | `nvidia-smi`가 GPU와 컴퓨트 기능을 보고하는지 |
| cuda-gdb        | 선택 사항. `cargo oxide debug`에만 필요  |
| compute-sanitizer | 선택 사항. `cargo oxide sanitize`에만 필요 |

libNVVM/nvJitLink/libdevice 검사는 커널이 CUDA libdevice 수학 함수(`sin`, `cos`, `exp`, `pow`, `sqrt` 등)를 호출할 때만 의미가 있습니다. 커널이 순수 산술만 수행한다면 세 검사가 실패해도 문제없습니다. 모두 CUDA Toolkit에 포함되므로 별도 다운로드는 필요하지 않습니다. 검사가 실패하면 doctor가 해당 구성 요소의 표준 설치 위치를 출력합니다.

doctor 자체에는 CUDA 툴킷이나 드라이버가 필요하지 않으며 사전 빌드도 하지 않으므로 아무것도 설치되지 않은 시스템에서도 동작합니다. 두 검사는 치명적 오류가 아니라 정보만 제공합니다. 코드 생성 백엔드의 `.so`가 없으면 `cargo oxide setup`을 실행하면 되며 `run`/`build`도 필요 시 빌드합니다. 드라이버/GPU는 `cargo oxide run`에만 필요하고 `build`와 `pipeline`은 GPU 없이 동작합니다.

## `cargo oxide inspect` — 생성된 PTX 확인

중간 덤프가 아니라 최종 디바이스 어셈블리만 필요하면 `inspect`를 사용하세요.

```bash
cargo oxide inspect vecadd
```

`inspect`는 `cargo oxide build`와 같은 방식으로 예제를 빌드한 뒤 생성된 PTX를 출력합니다. 백엔드가 무엇을 출력했는지 빠르게 확인할 때 사용하세요. MIR/LLVM dialect/`.ll` 단계도 필요하면 아래의 `pipeline`을 사용합니다. `inspect`는 PTX만 표시하므로 환경에서 비 PTX 출력 모드(`CUDA_OXIDE_MATERIALIZE_CUBIN`, `CUDA_OXIDE_EMIT_NVVM_IR`)가 활성화되어 있으면 오류와 함께 종료됩니다.

## `cargo oxide clean` — 로컬 빌드 산출물 제거

`clean`은 현재 워크스페이스나 독립 프로젝트의 프로젝트 로컬 Cargo `target/` 디렉터리와 생성된 cuda-oxide 디바이스 아티팩트(`.ptx`, `.ll`, `.opt.ll`, `.ltoir`, `.cubin`, `.target`/`.options`/`.cubin.target` 사이드카)를 삭제합니다. 심볼릭 링크인 `target/` 디렉터리나 아티팩트는 오류와 함께 삭제를 거부하며 `~/.cargo/cuda-oxide/`의 공유 코드 생성 백엔드 캐시는 **삭제하지 않습니다**.

```bash
cargo oxide clean
```

## `cargo oxide pipeline` — 컴파일 파이프라인 검사

커널이 오류 없이 잘못된 결과를 내면 전체 컴파일 파이프라인을 살펴 생성된 코드를 확인하세요.

```bash
cargo oxide pipeline vecadd
```

전체 파이프라인 출력을 표시합니다.

1. **MIR 수집** — 수집기가 찾은 함수
2. **`dialect-mir`** — Rust MIR을 모델링한 pliron IR(`mem2reg` 전후)
3. **LLVM dialect** — `pliron-llvm`이 제공하는 LLVM IR 모델링 pliron IR(`mir-lower` 후)
4. **텍스트 LLVM IR** — 직렬화된 `.ll` 파일
5. **최종 PTX** — 생성된 어셈블리

### 환경 변수

For more targeted inspection:

| 변수                       | 효과                            |
|:-------------------------------|:----------------------------------|
| `CUDA_OXIDE_VERBOSE=1`         | 상세 컴파일러 출력           |
| `CUDA_OXIDE_SHOW_RUSTC_MIR=1`  | 가져오기 전 rustc MIR 덤프  |
| `CUDA_OXIDE_DEBUG=line-tables` | 소스 라인 테이블 메타데이터 출력   |
| `CUDA_OXIDE_DEBUG=full`        | 기본 지역 변수와 인자에 대한 전체 메타데이터 출력 |

## Nsight Compute로 프로파일링

성능 디버깅을 위해 NVIDIA의 **Nsight Compute**(`ncu`)는 루프라인 분석, 메모리 처리량, 점유율 지표를 제공합니다.

```bash
ncu --set full ./target/release/my_example
```

cuda-oxide 커널은 `debug::prof_trigger::<N>()`으로 프로파일러 트리거를 출력할 수 있습니다. 이 함수는 `pmevent` 명령을 생성하며 Nsight Compute와 Nsight Systems가 타임라인 주석에 사용할 수 있습니다.

:::{seealso}
전체 프로파일링 도구 설명은 [Nsight Compute 문서](https://docs.nvidia.com/nsight-compute/)를 참조하세요.
:::

## 흔한 문제와 해결책

| 문제                          | 증상                                    | 해결 방법                                                              |
|:---------------------------------|:-------------------------------------------|:-----------------------------------------------------------------|
| 출력 버퍼 데이터 경합  | 결과 오류, 비결정적 동작           | `DisjointSlice`와 일치하는 준비 실행 형상을 사용      |
| `sync_threads()` 누락         | 공유 메모리에서 오래된 값 읽기                  | 쓰기와 읽기 사이에 배리어 추가                             |
| `shared_mem_bytes` 오류         | `LAUNCH_OUT_OF_RESOURCES` 또는 잘못된 데이터  | 실제 `DynamicSharedArray` 사용량에 맞춰 `LaunchConfig` 설정        |
| 원시 포인터 범위 초과  | 트랩 또는 조용한 손상                  | 범위 검사를 위해 `DisjointSlice::get_mut` 사용                 |
| 커널에서 `panic!("message")`    | 스레드 트랩, 메시지는 표시되지 않음        | GPU에는 패닉 런타임이 없습니다. 값을 보려면 검사 전에 `gpu_printf!` 사용 |
| 실행 후 동기화 누락  | 호스트가 오래된 데이터 읽기                      | `to_host_vec`, `stream.synchronize()`, `.sync()` 호출         |
| 잘못된 아키텍처용 PTX 빌드         | `NO_BINARY_FOR_GPU`                        | `cargo oxide build --arch sm_XX`로 다시 빌드                    |

```{figure} images/debug-workflow.svg
:align: center
:width: 100%

디버깅 결정 트리입니다. 커널 문제는 컴파일 오류, 런타임 디바이스 실패, 조용한 데이터 손상의 세 갈래로 나뉘며 각각 다른 진단 도구를 사용합니다. 아래에는 흔한 해결 방법을 표시했습니다.
```

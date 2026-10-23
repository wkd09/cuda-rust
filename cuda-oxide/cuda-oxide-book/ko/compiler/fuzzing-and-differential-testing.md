# 퍼징과 차등 테스트

cuda-oxide의 정상 경로는 예제로 쉽게 테스트할 수 있습니다. `vecadd`, `gemm`, `sharedmem` 등은 이미 알려진 프로그램이 계속 작동하는지 확인합니다. 하지만 컴파일러는 이미 아는 프로그램에서만 실패하지 않습니다. 손으로 작성할 생각을 못 했던 작고 기묘한 모서리 사례, 예를 들어 분기 뒤의 형 변환, 호출 종료자 안의 튜플, 아무도 사용하지 않을 것 같았던 정수 폭에서도 실패합니다.

이런 경우에 퍼징이 필요합니다.

cuda-oxide는 rustlantis 기반의 작은 하네스를 사용해 임의의 사용자 정의 MIR 프로그램을 생성하고, 일반 Rust CPU 백엔드와 cuda-oxide GPU 백엔드 양쪽에서 실행한 다음 중간값의 간결한 추적 결과를 비교합니다. 목표는 GPU에서 Rust 전체가 옳다는 것을 증명하는 게 아닙니다. 그보다 현실적이고 유용한 목표는 직접 작성하지 않은 프로그램으로 MIR 가져오기, 낮추기 파이프라인, LLVM 내보내기, PTX 생성, 런타임 실행을 압박 테스트하는 것입니다.

---

## 비교 대상

처음에는 “CPU와 GPU 실행을 비교한다”는 말이 의심스럽게 들릴 수 있습니다. CPU와 GPU의 실행 모델은 다릅니다. CPU는 스칼라 스레드 하나로 실행하지만 GPU는 수천 개의 SIMT 레인을 사용합니다. 제어 흐름 분기, 메모리 공간, 동기화 규칙도 다릅니다. 일반적인 CPU 대 GPU 의미 비교는 그럴듯한 방식으로 스스로를 속이는 일이 될 수 있습니다.

그래서 하네스는 그런 비교를 의도적으로 피합니다.

생성된 MIR을 스칼라 GPU 프로그램으로 실행합니다.```text
<<<1, 1>>>
```

블록 하나, 스레드 하나입니다. 스레드 간 통신도 스케줄링 변수도 없습니다. GPU는 병렬 프로그래밍 모델이 아니라 같은 스칼라 MIR을 위한 두 번째 코드 생성 대상으로 사용됩니다. 비교 경로는 다음과 같습니다.```text
same generated MIR
  -> normal rustc/LLVM CPU execution
  -> cuda-oxide -> LLVM IR -> PTX -> CUDA execution
  -> compare trace hash
```

해시가 같으면 생성된 프로그램이 양쪽 경로에서 동일한 중간값 시퀀스를 관찰한 것입니다. 해시가 다르면 cuda-oxide 경로의 어떤 부분을 조사해야 한다는 뜻입니다.```{note}
The trace is intentionally compact. Instead of copying every intermediate value
back to the host, each `dump_var(...)` call folds values into one `u64` using a
byte-wise hash. This makes generated tests cheap to run and easy to compare.
```

---

## 구성 요소

퍼징 설정은 네 부분으로 나뉩니다.

| 구성 요소 | 역할 |
|:---|:---|
| `cuda-oxide/crates/fuzzer` | 공유 추적 API와 저장소에 포함된 rustlantis |
| `cuda-oxide/crates/fuzzer/tools/mir_generator.py` | seed를 `generated_case.rs`로 변환하는 어댑터 |
| `cuda-oxide/crates/fuzzer/tools/run_seed.py` | 일괄 실행기와 산출물 기록기 |
| `rustlantis-smoke` | stable CPU/GPU 실행 하네스 |

`cuda-oxide/crates/fuzzer`는 일반 작업공간 크레이트지만 라이브러리 인터페이스는 `no_std`입니다. 장치 코드는 여기서 `trace_reset`, `trace_finish`, 제네릭 `dump_var`를 가져옵니다. rustlantis 소스는 `cuda-oxide/crates/fuzzer/rustlantis` 아래에 포함되어 있으며 Rust 라이브러리 의존성으로 쓰이지 않고 외부 생성기로 호출됩니다.

`rustlantis-smoke` 예제는 `cuda-oxide/crates/rustc-codegen-cuda/examples/` 아래에 있습니다. 호스트/GPU 실행 로직과 직접 작성한 작은 검증 테스트를 소유하며, 생성 파일 하나를 포함합니다.```text
cuda-oxide/crates/rustc-codegen-cuda/examples/rustlantis-smoke/src/generated_case.rs
```

퍼징 도구는 seed마다 이 파일을 다시 씁니다. 예제의 나머지는 그대로 유지됩니다. 덕분에 하네스를 쉽게 검토할 수 있습니다. seed에서 실패가 발생하면 생성된 MIR이 한곳에 분리되어 있습니다.

---

## Seed 처리 과정

하나의 seed는 다음 단계를 거칩니다.

1. **생성.** rustlantis가 숫자 seed와 작은 설정을 받습니다. seed와 설정이 같으면 같은 사용자 정의 MIR 프로그램이 생성됩니다.
2. **추출.** `mir_generator.py`가 처음 생성된 `#[custom_mir]` 함수를 가져옵니다.
3. **변환.** rustlantis는 `dump_var(a, b, c)` 같은 호출을 생성합니다. 사용자 정의 MIR 호출 피연산자는 튜플 표현식을 까다롭게 처리하므로 어댑터가 이를 튜플 로컬과 제네릭 `fuzzer::dump_var(...)` 호출 하나로 바꿉니다.
4. **삽입.** 변환된 함수와 작은 래퍼를 `src/generated_case.rs`에 기록합니다.
5. **실행.** `cargo oxide run rustlantis-smoke`가 CPU 기준 실행과 GPU 커널을 실행합니다.
6. **분류.** `run_seed.py`가 seed의 통과, 불일치, 컴파일 실패, 현재 어댑터 지원 범위 초과 여부를 기록합니다.

체크인된 생성 사례는 seed `19`를 사용합니다. 최종값 하나만이 아니라 여러 중간값을 덤프하는, 하네스가 확인하려는 핵심 특성을 보여주기 때문입니다.```rust
__rl_dump0 = (Move(_1), Move(_2), Move(_3), Move(_4));
Call(_9 = dump_var(Move(__rl_dump0)), ReturnTo(bb4), UnwindUnreachable())

__rl_dump1 = (Move(_6),);
Call(_9 = dump_var(Move(__rl_dump1)), ReturnTo(bb5), UnwindUnreachable())
```

따라서 최종 추적 해시에는 프로그램의 한 지점에서 얻은 여러 값과 이후 지점의 값이 함께 포함됩니다. 결과는 여전히 간결하지만 단순히 “반환값이 일치했는가?”만 검사하는 것은 아닙니다.

---

## 실행하기

seed 하나를 실행합니다.```bash
python3 cuda-oxide/crates/fuzzer/tools/run_seed.py --seed 192
```

여러 seed를 묶어 실행합니다.```bash
python3 cuda-oxide/crates/fuzzer/tools/run_seed.py --start 0 --count 20 --keep-going
```

유용한 플래그는 다음과 같습니다.

- `--keep-going`: seed 실패 후에도 계속 진행합니다.
- `--keep-logs`: 통과한 seed의 로그도 기록합니다.
- `--no-build`: 이미 빌드된 rustlantis 생성기를 재사용합니다.
- `--append-summary`: 기존 요약을 덮어쓰지 않고 뒤에 추가합니다.

기본적으로 실행 시작 시 `summary.jsonl`을 새로 씁니다. 따라서 “방금 실행한 결과가 무엇인가?”를 바로 확인할 수 있습니다. 이력을 유지하려면 `--append-summary`를 지정하세요.

---

## 결과 읽기

실행기는 seed마다 한 줄을 출력하고 마지막에 전체 요약을 표시합니다.```text
results:
  seed 0: UNSUPPORTED [adapter] unsupported return type for return-value tracing: *const i8 (...)
  seed 1: COMPILE_FAIL [backend] Unsupported construct: Type translation not yet implemented for: RigidTy(Char) (...)
summary: COMPILE_FAIL=1, UNSUPPORTED=1
```

상태는 다음을 의미합니다.

| 상태 | 의미 |
|:---|:---|
| `PASS` | CPU와 GPU의 추적 결과가 일치함 |
| `MISMATCH` | CPU와 GPU의 추적 결과가 다름 |
| `COMPILE_FAIL [backend]` | 어댑터가 사례를 생성했지만 cuda-oxide에서 실패함 |
| `UNSUPPORTED [adapter]` | rustlantis가 MIR을 생성했지만 어댑터가 지원하지 않음 |

`MISMATCH`는 가장 심각하게 다뤄야 할 결과입니다. 두 경로 모두 컴파일과 실행에 성공했지만 관찰한 값이 다릅니다. 백엔드 정확성 버그일 가능성이 있습니다.

`COMPILE_FAIL [backend]`는 생성 사례가 어댑터 단계를 통과해 cuda-oxide까지 도달했다는 뜻입니다. 현재 지원하지 않는 MIR 타입처럼 예상 가능한 실패일 수도 있지만, 거부한 구성요소는 백엔드입니다.

`UNSUPPORTED [adapter]`는 rustlantis가 프로그램을 만들었지만 어댑터가 이를 smoke 테스트 사례로 변환하지 않았다는 뜻입니다. 예를 들면 다음과 같습니다.```text
unsupported return type for return-value tracing: *const i8
```

이 거부는 덤프 지점이 아니라 함수 경계에서 발생합니다. rustlantis는 해시할 수 있는 값만 덤프하며 어댑터는 그 값을 모두 받아들입니다. 어댑터가 거부할 수 있는 것은 함수 자체입니다. 추적 API가 해시할 수 없는 반환 타입이거나 리터럴을 만들 수 없는 인자 타입일 수 있습니다. 추적기는 다음 스칼라를 해시합니다.```text
bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, char,
f32, f64
```

또한 위 목록에 있는 타입으로 구성된 배열과 튜플도 중첩 깊이에 관계없이 해시합니다. 내부 스칼라를 하나씩 해시하기 때문입니다. `TraceDump` 구현과 일치하도록 튜플은 최대 5개 원소까지 지원합니다.```text
[u128; 4]                  hashes fine
(u32, char)                hashes fine
(i8, i8, i8, i8, i8, i8)   rejected: six elements
*const i8                  rejected: pointer
```

지원하지 않는 항목은 원시 포인터와 참조입니다. CPU와 GPU는 같은 객체에 대해 서로 다른 주소를 보유하므로 포인터 해시가 일치할 수 없습니다. 원소가 5개를 초과하는 튜플도 제외됩니다. 따라서 어댑터 수준에서 지원하지 않는 많은 경우는 “잘못된 MIR”도 cuda-oxide 버그도 아닙니다. 포인터를 거부하는 것은 설계상 영구적인 제약입니다. 폭을 늘려도 해시가 일치하지 않습니다. 튜플 제한은 단지 구현 공백입니다. 여섯 원소 튜플을 해시하려면 `TraceDump` 구현 하나와 간식이 조금 더 필요합니다.

---

## 산출물

seed별 로그는 다음 디렉터리에 저장됩니다.```text
cuda-oxide/crates/fuzzer/artifacts/
```

실패 로그에는 다음 항목이 포함됩니다.

- seed
- 상태와 단계
- 원인
- 반환 코드
- 명령
- 명령 전체 출력
- 생성된 `generated_case.rs` 스냅샷(있는 경우)

생성 스냅샷은 중요합니다. CI나 장시간 일괄 실행에서 백엔드 실패가 발생하면 로그만으로 원인이 된 정확한 MIR 형태의 프로그램을 확인할 수 있습니다. seed를 사용해 재생성할 수도 있지만 스냅샷이 있으면 왕복 과정이 필요 없습니다.

---

## 현재 한계

현재 설정은 의도적으로 작습니다. 한 번에 모든 Rust 구문을 지원하기보다 스칼라 사용자 정의 MIR과 백엔드 연결에 첫 단계를 집중합니다. 그래서 초기 seed 중 상당수가 `UNSUPPORTED [adapter]`로 분류됩니다.

지원 범위 확장은 점진적으로 진행하며 두 단계가 이미 반영되었습니다. 어댑터는 이제 추적 API가 해시할 수 있는 모든 스칼라를 받습니다(마지막으로 f32와 f64가 [#484](https://github.com/NVlabs/cuda-oxide/pull/484)에서 추가됨). 이 스칼라로 만든 배열과 튜플도 지원합니다([#792](https://github.com/NVlabs/cuda-oxide/pull/792)). 남은 항목은 다음과 같습니다.

1. cuda-oxide 타입 변환에서 `char`를 지원할지, 지원한다면 어떻게 할지 결정합니다.
2. 제어 흐름과 형 변환 범위를 확장합니다.
3. 구조체와 열거형을 추가합니다. 현재 추적기는 내부를 들여다볼 수 없습니다.
4. 실패 seed 최소화 기능을 추가합니다.

이 순서는 의도된 것입니다. 첫날부터 모든 것을 생성하는 퍼저는 대부분 잡음을 냅니다. 한 축씩 범위를 넓히는 퍼저는 무엇이 왜 깨졌는지 알려줍니다. 훨씬 다루기 쉽습니다.

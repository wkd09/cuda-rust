# 카탈로그 생성 인트린식

cuda-oxide의 디바이스 인트린식은 컴파일러 단계를 각각 직접 고쳐 구현하기보다, 검토된 입력 카탈로그에 정의하고 `cuda-intrinsics-gen`으로 API, dialect, importer, lowering, 대상, 참조, probe 코드를 생성합니다. 카탈로그가 모델링할 수 없는 전용 검증 또는 lowering이 필요하면 [새 인트린식 추가](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/adding-new-intrinsics.html)를 참고하세요.

## 먼저 올바른 경로 선택

```text
기존 카탈로그 패밀리가 같은 의미와 lowering 형태를 표현하는가?
    ├── 예: 해당 overlay 패밀리와 ABI ledger를 확장
    └── 아니요: 새 패밀리/형태 또는 수작업 구현이 필요한지 판단
```

LLVM 선언만으로 인트린식을 허용할 수는 없습니다. 카탈로그에는 Rust API, 안전성·부수 효과, 예상 PTX, 대상 요구 사항, ABI 정체성, lowering 근거 등 프로젝트 정책이 포함됩니다. 일반 기여에서는 생성 파일이나 `intrinsics/catalog.json`을 직접 고치지 말고 기존 검토 패밀리를 확장합니다.

## 기준 입력

```text
intrinsics/upstream.lock -> intrinsics/imported.json ----\
intrinsics/overlay.toml -> intrinsics/overlay/*.toml -----+
intrinsics/abi-v1.toml -----------------------------------+--> resolve
intrinsics/evidence/*.json -------------------------------/       |
                                                                  v
                                                     intrinsics/catalog.json
                                                            GENERATED
                                                                  |
                                                                  v
                                                        generated outputs
```

| 파일 | 역할 |
|:--|:--|
| `intrinsics/upstream.lock` | 가져온 LLVM 사실의 출처와 근거 probe에 사용할 rustc 커밋을 고정합니다. |
| `intrinsics/imported.json` | 고정 LLVM/NVPTX 소스의 선언을 기록합니다. 일반 추가는 이 파일을 사용하며 추출을 다시 실행하지 않습니다. |
| `intrinsics/overlay.toml` | 사용할 overlay 조각을 색인합니다. |
| `intrinsics/overlay/*.toml` | 검토된 인트린식 패밀리의 cuda-oxide 정책입니다. |
| `intrinsics/abi-v1.toml` | 안정 ABI ledger입니다. 끝에만 추가합니다. |
| `intrinsics/evidence/*.json` | 필요한 경우 검토된 lowering 근거입니다. |
| `intrinsics/catalog.json` | 해석된 생성 계획입니다. 직접 편집하지 마세요. |

생성 소스와 probe/참조 산출물에는 `DO NOT EDIT` 표시가 있습니다. 원본 입력을 바꾸고 다시 생성합니다.

### 카탈로그 메타데이터

Overlay와 결과 카탈로그의 `schema`는 서로 다른 형식 버전입니다.

```text
intrinsics/overlay.toml   schema = 44   (overlay 입력 형식)
        | cuda-intrinsics-gen
        v
intrinsics/catalog.json   "schema": 46  (직렬화 형식)
```

`overlay.toml`의 `schema`는 `OVERLAY_SCHEMA`와 일치해야 하며 resolver가 불일치를 거부합니다. `catalog_version`은 검토 계약, `intrinsic_abi`는 안정 ABI 세대, `backend_profile`은 lowering 근거 해석용 프로필입니다. 생성 JSON에는 이 값을 직접 작성하지 마세요. JSON 헤더의 `schema`는 `CATALOG_SCHEMA` 형식 버전이고 `generator_version`은 파일의 생성 배너에도 표시되는 생성기 패키지 버전입니다.

## 일반 절차: 기존 패밀리 확장

### 1. 가까운 형제 항목 찾기

생성 Rust 코드나 `intrinsics/imported.json` 전체에서 시작하지 말고 `intrinsics/overlay/`에서 같은 패밀리와 lowering 형태를 찾습니다. 필요한 upstream 필드만 가져온 사실과 대조합니다.

```bash
grep -R 'id = "<nearest-sibling>"' intrinsics/overlay
rg '<llvm-or-record-name>' intrinsics/imported.json
```

검증된 항목을 바탕으로 하면 검토 범위와 예상 밖 생성 diff가 줄어듭니다.

### 2. overlay 추가 또는 확장

`intrinsics/overlay/<family>.toml`을 수정합니다. 생성 연산, importer dispatch, lowering 표, `intrinsics/catalog.json`은 직접 편집하지 마세요. 새 조각은 `intrinsics/overlay.toml`의 정렬된 `shards` 목록에도 등록해야 합니다. `generate`와 `check`가 색인 누락을 거부합니다.

### 3. ABI ledger에 추가

새 안정 ID를 `intrinsics/abi-v1.toml` 끝에 추가합니다. 기존 항목은 재정렬·삭제·번호 변경·재사용하지 않습니다. 카탈로그 ID, 연산 키, Rust 시그니처는 overlay와 일치해야 합니다. 선행 PR이 ledger를 갱신했다면 최신 `main`으로 rebase하고 새 끝에 이어 붙입니다.

### 4. 필요한 경우 lowering 근거 수집

기존 검토 형태면 형제 항목 계약을 따릅니다. 새 경로의 후보 근거가 필요할 때 명시적으로 probe합니다.

```bash
HOST="$(rustc -vV | sed -n 's/^host: //p')"
LLC="$(rustc --print sysroot)/lib/rustlib/$HOST/bin/llc"

cargo run -p cuda-intrinsics-gen -- probe --candidate \
  --intrinsic <catalog-id> \
  --llc "$LLC" \
  --gpu-target <sm-target> \
  --ptx-feature <ptx-feature> \
  --ptxas <path-to-ptxas>
```

터미널 어셈블러를 사용할 수 없으면 `--ptxas` 대신 `--skip-terminal`을 명시합니다. 둘 중 하나는 필수이며 자동 대체하지 않습니다. 후보 결과를 검토한 뒤에만 계약에 반영합니다.

### 5. 재생성 및 점검

```bash
cargo run -p cuda-intrinsics-gen -- generate
git status --short
git diff --stat
git diff -- intrinsics cuda-oxide/crates/cuda-intrinsics cuda-oxide/crates/cuda-device \
  cuda-oxide/crates/dialect-nvvm cuda-oxide/crates/mir-importer cuda-oxide/crates/mir-lower \
  cuda-oxide/crates/cuda-oxide-codegen cuda-oxide/crates/rustc-codegen-cuda
```

새 파일은 stage 전까지 untracked라 `git diff`에 보이지 않으므로 상태도 확인하세요. 기존 패밀리의 생성 변경은 형제 형태를 따라야 합니다. 무관한 변경이 있으면 overlay, ledger, 근거, 기준 브랜치를 다시 점검합니다.

## 푸시 전 검증

```bash
just -f cuda-oxide/Justfile check-intrinsics upstream/main
```

실제 PR 기준 브랜치가 다르면 바꿔 사용합니다. recipe는 다음 CI 검사 계약을 실행합니다.

```text
cuda-intrinsics-gen check
cuda-intrinsics-gen probe --all --skip-terminal --per-target
cuda-intrinsics-gen check-abi-history --base-ref <base>
```

이 명령을 일반 인터페이스로 사용하세요. `check-abi-history`는 기준 ref까지 이력을 탐색하므로 전체 clone이 필요합니다(CI는 `fetch-depth: 0`). 생성기는 `rustfmt`를 사용하며 도구 체인 파일이 구성 요소를 고정합니다. 단일 인트린식 확인은 다음과 같습니다.

```bash
cargo run -p cuda-intrinsics-gen -- probe \
  --intrinsic <catalog-id> \
  --skip-terminal
```

책도 바꿨다면 다음을 실행합니다.

```bash
just -f cuda-oxide/Justfile book
bash cuda-oxide/scripts/check-book-api-names.sh
git diff --check
```

## 기존 카탈로그로 표현할 수 없는 경우

새 lowering 형태, 피연산자 어댑터, 결과 계약, 패밀리 형태가 필요하면 이는 단순 등록이 아닙니다. `cuda-intrinsics-gen/src/model/`, `resolve/`, `render/` 및 필요에 따라 `cuda-oxide/crates/mir-lower/`나 `dialect-nvvm/`의 생성기·컴파일러를 수정해야 합니다. 가능하면 일반 카탈로그 등록과 분리하되 ABI ledger와 검증은 적용합니다. 설계상 수작업으로 남길 연산은 [새 인트린식 추가](https://nvidia.github.io/cuda-rust/cuda-oxide/compiler/adding-new-intrinsics.html)를 따릅니다.

## 일반 추가에서 추출하지 않기

`extract`는 LLVM 소스나 고정 버전을 갱신할 때 가져온 정보를 새로 만들기 위한 명령입니다. 이미 `intrinsics/imported.json`에 있는 인트린식의 일반 추가에는 실행하지 않습니다. 커밋된 입력을 사용하면 LLVM checkout에 의존하지 않습니다.

## 직접 편집하지 말 것

다음 생성 산출물을 직접 수정하지 마세요.

- `intrinsics/catalog.json`
- `cuda-intrinsics/src/generated/`, `cuda-device/src/generated/`, `dialect-nvvm/src/ops/generated/`의 파일
- `mir-importer`의 생성 dispatch와 `mir-lower`의 생성 변환 표
- 생성 대상/collector 메타데이터 및 probe/참조 결과

재생성으로 변경이 덮어써진다면 수정 계층이 잘못된 것입니다. 진실의 원천인 검토 입력, ABI, 근거 또는 생성기 로직을 수정하세요.

:orphan:

# cuda-oxide 책

이 디렉터리에는 cuda-oxide 책의 한국어판 소스가 있습니다. 책을 빌드하려면 저장소 루트에서 다음 명령을 실행합니다.

```text
make setup
source .venv/bin/activate
make livehtml
```

기본 설정에서는 `http://127.0.0.1:8000/`에서 책을 확인할 수 있습니다.

## 관련 문서

- **API 문서**: 프로젝트 루트에서 `cargo doc --open`을 실행합니다.
- **예제**: `cuda-oxide/crates/rustc-codegen-cuda/examples/`를 참고하세요.

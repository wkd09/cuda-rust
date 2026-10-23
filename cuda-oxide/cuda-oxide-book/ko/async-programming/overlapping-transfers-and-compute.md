# 전송과 계산 중첩하기

페이지 가능한(pageable) 호스트 메모리는 편리하지만 CUDA가 DMA 전송 전에 내부 페이지 고정 메모리로 데이터를 옮길 수 있습니다. 이 추가 스테이징 작업은 비동기 복사를 제한합니다. `PinnedHostBuffer<T>`를 사용하면 CUDA가 안정적인 호스트 주소에서 직접 전송할 수 있습니다. 복사와 계산을 실제로 중첩하려면 기본 스트림이 아닌 스트림에서 비동기 복사도 사용해야 합니다.

전체 예제를 실행합니다.

```bash
cargo oxide run pinned_overlap
```

이 예제는 간단한 제자리 증가 커널을 사용하므로 실행 시간은 계산보다 데이터 이동의 영향을 더 많이 받습니다.

## 복사 API 선택

안전한 헬퍼는 반환 전에 스트림을 동기화합니다. 복사가 진행되는 동안 호스트에서 다른 작업을 할 필요가 없을 때 사용합니다.

```rust
device.copy_to_pinned_host(&stream, &mut host)?;
// 헬퍼가 stream을 동기화했으므로 host를 안전하게 읽을 수 있습니다.
```

비동기 헬퍼는 복사가 끝나기 전에 반환합니다. 동기화와 버퍼 수명은 호출자가 관리합니다.

```rust
// SAFETY: done이 발생할 때까지 host와 device가 살아 있고 재사용되지 않습니다.
unsafe { device.copy_to_pinned_host_async(&stream, &mut host)? };
let done = stream.record_event(None)?;
```

호출자는 두 버퍼를 모두 살아 있게 유지해야 합니다. 전송 중인 호스트 버퍼를 읽거나 수정해서는 안 되며, 버퍼를 해제하거나 재사용하기 전에 스트림 작업이 끝날 때까지 기다려야 합니다. `copy_from_pinned_host_async`에도 같은 규칙이 적용됩니다. 장치 버퍼를 미리 할당해 다시 채우세요. `from_pinned_host`는 복사를 큐에 넣기 전에 동기식 장치 할당을 수행합니다.

## 고정 스테이징 버퍼 순환 사용

하나의 스트림 안에서는 작업 순서가 보장됩니다. 동시 복사와 계산을 지원하는 하드웨어에서는 독립 스트림이 동시에 진행될 수 있습니다.

```text
stream 0: upload 0 -> kernel 0 -> download 0 -> upload 3 -> kernel 3
stream 1: upload 1 -> kernel 1 -> download 1 -> upload 4 -> kernel 4
stream 2: upload 2 -> kernel 2 -> download 2 -> upload 5 -> kernel 5
```

예제는 각 스트림에 지속적으로 재사용하는 `DeviceBuffer`, 고정 스테이징 버퍼, 완료 이벤트를 하나씩 둡니다. 업로드, 커널, 다운로드 순서는 스트림이 이미 보장하므로 이벤트는 호스트 측 슬롯 재사용을 보호할 때만 필요합니다. 슬롯을 재사용하기 전에 호스트가 이벤트를 기다리고, 완료된 결과를 꺼낸 뒤 스테이징 버퍼를 새 데이터로 채웁니다. 마지막에 모든 스트림을 합류시켜 CUDA 소유 메모리를 해제하기 전에 작업을 모두 비웁니다.

예제 루프에서 주변 설정과 결과 처리를 생략한 핵심 큐 등록 순서는 다음과 같습니다.

```rust
// SAFETY: 완료 전까지 슬롯 버퍼는 살아 있고 재사용되지 않습니다.
unsafe {
    devices[slot].copy_from_pinned_host_async(&streams[slot], &stagers[slot])?;
    module.increment(&streams[slot], config, &mut devices[slot])?;
    devices[slot].copy_to_pinned_host_async(&streams[slot], &mut stagers[slot])?;
}
completions[slot] = Some(streams[slot].record_event(None)?);
```

전체 설정과 슬롯 재사용 대기, 마지막 동기화는 [`run_overlapped` 구현](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap/src/main.rs)을 참조하세요.

## 측정 결과

대역폭 벤치마크는 세 번 워밍업한 뒤 CUDA 이벤트로 열 번 측정하고 중앙값을 보고합니다. 파이프라인 벤치마크는 32 MiB 청크 여덟 개의 호스트 벽시계 시간을 측정합니다. 직렬 경로는 각 청크마다 페이지 가능한 장치 버퍼를 할당하고 동기화합니다. 중첩 경로는 장치 버퍼와 고정 스테이징 버퍼 세 개를 미리 할당하며, 호스트 스테이징 복사, 이벤트 대기, 결과 복사 시간도 벽시계 측정에 포함합니다. 따라서 순수 복사 엔진 성능이 아니라 종단 간 경로 비교입니다.

기준 RTX 4090(`sm_89`) 실행 결과입니다.

| 파이프라인 | 시간 |
|---|---:|
| 페이지 가능한 메모리 직렬 처리 | 174.996 ms |
| 고정 메모리 중첩 처리 | 99.332 ms |
| 속도 향상 | 1.76배 |

전송 전체 표는 [예제 README](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap)에 있습니다. 결과는 GPU, PCIe 토폴로지, 드라이버, 시스템 부하에 따라 달라집니다.

## 실제 적용 시 고려사항

메모리를 페이지 고정하면 제한된 호스트 메모리를 사용하며, 페이지 가능한 입력을 스테이징 버퍼로 복사하는 작업도 공짜가 아닙니다. 작은 버퍼 링을 재사용하고 스트림 수를 늘리기 전에 복사 엔진을 프로파일링하세요.

낮은 수준의 스트림 모델은 [스케줄링과 스트림](https://nvidia.github.io/cuda-rust/cuda-oxide/async-programming/scheduling-and-streams.html)을, 실행 가능한 전체 구현은 [`pinned_overlap` 예제](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap)를 참조하세요.

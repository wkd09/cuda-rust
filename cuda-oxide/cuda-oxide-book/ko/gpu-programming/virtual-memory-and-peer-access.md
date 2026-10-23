# 가상 메모리 관리와 피어 액세스

CUDA 가상 메모리 관리(VMM)는 물리 메모리 할당, 가상 주소 예약, 매핑, 접근 권한 설정을 서로 독립된 작업으로 분리합니다.

메모리를 할당하고 바로 사용할 수 있는 디바이스 포인터를 반환하는 `DeviceBuffer`보다
저수준 API입니다. 애플리케이션에서 물리 메모리를 어디에 매핑할지 또는 어떤 GPU가 접근할 수 있을지 명시적으로 제어해야 할 때 VMM이 유용합니다.



피어 투 피어 액세스(P2P)는 관련은 있지만 별개의 메커니즘입니다. 지원되는 NVLink 또는 PCIe 같은 인터커넥트를 통해 한 CUDA 컨텍스트가 다른 GPU에 연결된 메모리에 직접 접근할 수 있게 합니다.



이 장에서는 다음 내용을 다룹니다.

* 두 GPU가 피어 액세스를 지원하는지 조회하기
* 방향별 피어 액세스 활성화 및 비활성화
* VMM을 사용한 물리 디바이스 메모리 할당
* 가상 주소 범위 예약
* 예약한 범위에 물리 메모리 매핑
* 하나 이상의 디바이스에 매핑 접근 권한 부여
* 하나의 물리 할당을 여러 GPU 가시 주소 범위에 매핑
* 동기화와 리소스 해제 순서

:::{seealso}
전체 통합 테스트는 다음 파일에서 볼 수 있습니다.
[`cuda-core/tests/simt_vmm_p2p.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-core/tests/simt_vmm_p2p.rs).

관련 구현은 다음과 같습니다.

* [`cuda-core/src/simt/vmm.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-core/src/simt/vmm.rs)
* [`cuda-core/src/simt/peer.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-core/src/simt/peer.rs)
:::

## VMM과 P2P가 해결하는 서로 다른 문제

VMM은 물리 메모리와 가상 주소 사이의 관계를 제어합니다.

```text
물리 메모리 할당
        |
        v
가상 주소 예약
        |
        v
Mapping
        |
        v
디바이스별 접근 권한
```

P2P는 CUDA 컨텍스트가 다른 디바이스에 연결된 메모리에 접근할 수 있는지 제어합니다.


```text
GPU 0의 컨텍스트
        |
        | enable_peer_access(&gpu0, &gpu1)
        v
GPU 1을 통해 접근 가능한 메모리
```

멀티 GPU 애플리케이션에서는 두 메커니즘을 모두 사용할 수 있습니다.

1. 한 디바이스에 물리 메모리를 할당합니다.
2. 가상 주소 범위를 예약합니다.
3. 해당 범위에 물리 할당을 매핑합니다.
4. 두 디바이스에 해당 매핑의 접근 권한을 부여합니다.
5. 필요한 방향으로 피어 액세스를 활성화합니다.

한 작업을 수행해도 다른 작업이 자동으로 이루어지지는 않습니다. VMM 매핑에는 여전히 명시적인 접근 권한이 필요하며, 컨텍스트가 피어 컨텍스트의 메모리에 직접 접근해야 한다면 피어 액세스를 활성화해야 합니다.



## 하드웨어 및 토폴로지 요구 사항

피어 액세스 가능 여부는 실제 GPU 토폴로지, CUDA 드라이버, 플랫폼 설정에 따라 달라집니다. 같은 컴퓨터에 있는 두 GPU라고 해서 반드시 서로 직접 접근할 수 있는 것은 아닙니다.



피어 액세스를 활성화하기 전에 항상 지원 여부를 조회하세요.

```rust
use cuda_core::simt::peer;
use cuda_core::{CudaContext, DriverError};

fn check_peer_access() -> Result<(), DriverError> {
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if peer::can_access_peer(&ctx0, &ctx1)? {
        println!("GPU 0 can access GPU 1");
    } else {
        println!("GPU 0 cannot access GPU 1");
    }

    Ok(())
}
```

`can_access_peer(from, to)`는 기능 지원 여부만 조회합니다. 어느 컨텍스트도 변경하지 않습니다.


:::{note}
피어 액세스는 방향성이 있습니다. GPU 0에서 GPU 1로의 지원 여부를 반대 방향도 활성화되었다는 증거로 간주하면 안 됩니다.

:::

## 피어 액세스 활성화

`peer::enable_peer_access`를 사용하면 한 컨텍스트에서 다른 디바이스에 연결된 메모리에 접근할 수 있습니다.


```rust
use cuda_core::simt::peer;
use cuda_core::{CudaContext, DriverError};

fn enable_bidirectional_peer_access() -> Result<(), DriverError> {
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if !peer::can_access_peer(&ctx0, &ctx1)? {
        return Ok(());
    }

    peer::enable_peer_access(&ctx0, &ctx1)?;
    peer::enable_peer_access(&ctx1, &ctx0)?;

    // Both contexts may now initiate peer accesses.

    peer::disable_peer_access(&ctx0, &ctx1)?;
    peer::disable_peer_access(&ctx1, &ctx0)?;

    Ok(())
}
```

방향은 인자 순서로 결정됩니다.

| 호출                               | 결과                                  |
| :--------------------------------- | :-------------------------------------- |
| `enable_peer_access(&ctx0, &ctx1)` | GPU 0 컨텍스트에서 GPU 1 메모리에 접근 가능 |
| `enable_peer_access(&ctx1, &ctx0)` | GPU 1 컨텍스트에서 GPU 0 메모리에 접근 가능 |
| 두 호출 모두                         | 양방향 피어 액세스               |

이미 활성화된 상태에서 `enable_peer_access`를 호출해도 성공합니다.
마찬가지로 현재 비활성화된 액세스를 끄는 작업도 성공적인 무연산으로 처리됩니다.

:::{warning}
피어 메모리를 사용하는 작업을 모두 마친 다음 피어 액세스를 비활성화하세요.
커널이나 복사 작업이 피어 가시 메모리를 사용하는 동안 액세스를 끄면 해당 작업이 무효화될 수 있습니다.
:::

## VMM 수명 주기

VMM 할당은 다음 네 단계로 진행됩니다.

1. 디바이스의 할당 단위를 조회합니다.
2. 물리 메모리를 할당합니다.
3. 가상 주소 범위를 예약하고 그 안에 물리 메모리를 매핑합니다.
4. 하나 이상의 디바이스에 매핑 접근 권한을 부여합니다.

주요 API는 다음과 같습니다.

| API                                         | 용도                                   |
| :------------------------------------------ | :---------------------------------------- |
| `vmm::allocation_granularity(device)`       | 필요한 할당 단위 조회 |
| `vmm::align_size(size, granularity)`        | 크기를 해당 할당 단위에 맞춤          |
| `PhysicalAllocation::new(device, size)`     | 물리 메모리 할당                  |
| `VirtualReservation::new(size, alignment)`  | 가상 주소 범위 예약           |
| `Mapping::new(va, size, &physical, offset)` | VA 범위에 물리 메모리 매핑       |
| `vmm::set_access(va, size, devices)`        | 매핑된 범위에 디바이스 접근 권한 부여  |

## 할당 단위

CUDA VMM 작업에서 할당 크기와 매핑 크기는 디바이스 할당 단위의 배수여야 합니다.

할당 전에 `allocation_granularity`와 `align_size`를 사용하세요.

```rust
use cuda_core::simt::vmm;
use cuda_core::{CudaContext, DriverError};

fn aligned_allocation_size(requested: usize) -> Result<usize, DriverError> {
    let ctx = CudaContext::new(0)?;
    let granularity = vmm::allocation_granularity(ctx.cu_device())?;

    Ok(vmm::align_size(requested, granularity))
}
```

4KiB 같은 페이지 크기가 CUDA VMM의 할당 단위 요구 사항을 충족한다고 가정하지 마세요. 필요한 값은 디바이스마다 다릅니다.

## 단일 GPU VMM 매핑

다음 예제는 물리 메모리를 할당하고 가상 주소 범위에 매핑한 뒤, 매핑을 통해 데이터를 쓰고 다시 읽어옵니다.

```rust
use cuda_core::simt::vmm;
use cuda_core::{CudaContext, DriverError};

fn single_gpu_vmm_roundtrip() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let granularity = vmm::allocation_granularity(ctx.cu_device())?;
    let allocation_size = vmm::align_size(4096, granularity);

    // Stage 1: allocate physical memory.
    let physical =
        vmm::PhysicalAllocation::new(ctx.cu_device(), allocation_size)?;

    // Stage 2: reserve virtual address space.
    let reservation =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    // Stage 3: map the physical allocation into the reserved range.
    let mapping = vmm::Mapping::new(
        reservation.base(),
        allocation_size,
        &physical,
        0,
    )?;

    // Stage 4: grant GPU 0 read/write access to the mapping.
    vmm::set_access(
        reservation.base(),
        allocation_size,
        &[ctx.cu_device()],
    )?;

    let input: Vec<u8> = (0..64).map(|i| (i * 3 + 7) as u8).collect();

    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            reservation.base(),
            input.as_ptr(),
            input.len(),
            stream.cu_stream(),
        )?;
    }

    ctx.synchronize()?;

    let mut output = vec![0u8; input.len()];

    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            reservation.base(),
            output.len(),
            stream.cu_stream(),
        )?;
    }

    ctx.synchronize()?;

    assert_eq!(output, input);

    // Resources must be destroyed in dependency order.
    drop(mapping);
    drop(reservation);
    drop(physical);

    Ok(())
}
```

`Mapping::new`만으로는 매핑된 주소에 접근할 수 없습니다. `vmm::set_access`가 하나 이상의 디바이스에 접근 권한을 부여하기 전까지 매핑은 사용할 수 없습니다.

## 하나의 물리 할당을 두 GPU에 매핑

VMM을 사용하면 하나의 물리 할당을 여러 가상 주소 범위에 매핑할 수 있습니다.

다음 예제는 이러한 방식으로 동작합니다.

* GPU 0에 물리 메모리를 할당합니다.
* GPU 0용 가상 주소 매핑을 만듭니다.
* GPU 1용 가상 주소 매핑을 하나 더 만듭니다.
* 두 디바이스 모두 두 매핑에 접근할 수 있도록 권한을 부여합니다.
* GPU 0 매핑을 통해 데이터를 씁니다.
* GPU 1 매핑을 통해 같은 물리 메모리에서 데이터를 읽습니다.

```rust
use cuda_core::simt::{peer, vmm};
use cuda_core::{CudaContext, DriverError, IntoResult};
use std::mem::MaybeUninit;

fn gpu_count() -> Result<usize, DriverError> {
    unsafe {
        cuda_core::init(0)?;
    }

    let mut count = MaybeUninit::uninit();

    unsafe {
        cuda_core::sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

fn cross_gpu_vmm_roundtrip() -> Result<(), DriverError> {
    if gpu_count()? < 2 {
        println!("This example requires at least two CUDA devices.");
        return Ok(());
    }

    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if !peer::can_access_peer(&ctx0, &ctx1)? {
        println!("The selected devices do not support peer access.");
        return Ok(());
    }

    peer::enable_peer_access(&ctx0, &ctx1)?;
    peer::enable_peer_access(&ctx1, &ctx0)?;

    let granularity =
        vmm::allocation_granularity(ctx0.cu_device())?;
    let allocation_size =
        vmm::align_size(4096, granularity);

    // The physical memory is allocated only once, on GPU 0.
    let physical =
        vmm::PhysicalAllocation::new(
            ctx0.cu_device(),
            allocation_size,
        )?;

    // GPU 0 virtual address range.
    let reservation0 =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    let mapping0 = vmm::Mapping::new(
        reservation0.base(),
        allocation_size,
        &physical,
        0,
    )?;

    vmm::set_access(
        reservation0.base(),
        allocation_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )?;

    // GPU 1 virtual address range backed by the same physical allocation.
    let reservation1 =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    let mapping1 = vmm::Mapping::new(
        reservation1.base(),
        allocation_size,
        &physical,
        0,
    )?;

    vmm::set_access(
        reservation1.base(),
        allocation_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )?;

    let input: Vec<u32> = (0..256).collect();
    let byte_length =
        input.len() * std::mem::size_of::<u32>();

    let stream0 = ctx0.default_stream();
    ctx0.bind_to_thread()?;

    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            reservation0.base(),
            input.as_ptr(),
            byte_length,
            stream0.cu_stream(),
        )?;
    }

    ctx0.synchronize()?;

    let mut output = vec![0u32; input.len()];
    let stream1 = ctx1.default_stream();
    ctx1.bind_to_thread()?;

    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            reservation1.base(),
            byte_length,
            stream1.cu_stream(),
        )?;
    }

    ctx1.synchronize()?;

    assert_eq!(output, input);

    // Destroy mappings before their reservations and backing allocation.
    drop(mapping1);
    drop(mapping0);
    drop(reservation1);
    drop(reservation0);
    drop(physical);

    peer::disable_peer_access(&ctx0, &ctx1)?;
    peer::disable_peer_access(&ctx1, &ctx0)?;

    Ok(())
}
```

## 가상 주소가 일치한다는 보장은 없음

`VirtualReservation::new(size, 0)`은 CUDA 드라이버가 가상 주소를 선택하도록 요청합니다. 따라서 동일한 물리 할당을 기반으로 하는 두 예약도 서로 다른 기본 주소를 가질 수 있습니다.

```text
GPU 0 VA: 0x0000_7f10_0000_0000
                         \
                          same physical allocation
                         /
GPU 1 VA: 0x0000_7e80_0000_0000
```

현재 API는 물리 메모리 공유를 지원하지만, 모든 GPU에서 같은 숫자 값의 가상 주소로 해당 메모리에 접근할 수 있다고 보장하지는 않습니다.

코드는 현재 액세스 경로에 해당하는 예약에서 반환된 `base()`를 사용해야 합니다.

:::{warning}
이 방식을 완전한 대칭 힙이라고 설명하지 마세요. 대칭 힙을 사용하려면 참여하는 모든 디바이스에서 동등한 가상 주소를 확보해야 하지만, 현재 드라이버 선택 방식의 예약 API는 이를 보장하지 않습니다.
:::

## 소유권과 해제 순서

VMM 래퍼 타입은 RAII를 통해 CUDA 리소스를 소유합니다.

| 타입                 | CUDA 리소스                      | Drop 작업     |
| :------------------- | :--------------------------------- | :----------------- |
| `PhysicalAllocation` | 일반 물리 할당 핸들 | `cuMemRelease`     |
| `VirtualReservation` | 예약된 가상 주소 범위     | `cuMemAddressFree` |
| `Mapping`            | 물리 메모리와 가상 주소 간 매핑        | `cuMemUnmap`       |

리소스 간 의존 관계는 다음과 같습니다.

```text
PhysicalAllocation   VirtualReservation
        ^                    ^
         \                  /
          \                /
               Mapping
```

`Mapping` 하나는 `PhysicalAllocation` 하나와 `VirtualReservation` 하나를 사용하므로 두 리소스가 모두 매핑보다 오래 살아 있어야 합니다.

리소스는 역순으로 해제하세요.

```rust
ctx0.synchronize()?;
ctx1.synchronize()?;

drop(mapping1);
drop(mapping0);

drop(reservation1);
drop(reservation0);

drop(physical);
```

멀티 GPU 작업의 전체 정리 순서는 다음과 같습니다.

1. 매핑을 사용하는 작업 제출을 중단합니다.
2. 해당 매핑을 사용하는 모든 스트림과 컨텍스트를 동기화합니다.
3. 모든 `Mapping` 값을 Drop합니다.
4. 모든 `VirtualReservation` 값을 Drop합니다.
5. `PhysicalAllocation`을 Drop합니다.
6. 더 이상 필요하지 않다면 피어 액세스를 비활성화합니다.

:::{warning}
Rust 타입 시스템은 매핑 의존 관계를 표현하지 않습니다. `Mapping`은 전체 수명 동안 `PhysicalAllocation`이나 `VirtualReservation`을 빌리지 않습니다. 필요한 해제 순서를 지키는 책임은 애플리케이션에 있습니다.
:::

## VMM 포인터를 소유 버퍼로 감싸지 마세요

VMM 가상 주소의 소유권은 해당 `Mapping`, `VirtualReservation`, `PhysicalAllocation`에 있습니다.

API가 비소유 변환을 명시적으로 정의하지 않는 한, 해당 포인터로 소유형 `DeviceBuffer`를 만들지 마세요. 소유형 버퍼는 `cuMemFree`로 포인터를 해제하려 할 수 있지만, VMM 리소스는 `cuMemUnmap`, `cuMemAddressFree`, `cuMemRelease`를 사용해야 합니다.

원시 `CUdeviceptr`는 비소유 디바이스 주소를 받는 API에만 전달하세요.

## 오류 처리와 미지원 시스템

이식성 있는 멀티 GPU 애플리케이션에서는 다음 상황을 각각 처리해야 합니다.

| 상황                         | 권장 동작                      |
| :-------------------------------- | :---------------------------------------- |
| 디바이스가 두 개 미만            | P2P 경로 건너뛰기                         |
| `can_access_peer`가 `false` 반환 | 호스트 경유 전송 또는 다른 경로 사용 |
| VMM 할당 실패              | CUDA 드라이버 오류 보고              |
| 접근 권한 설정 실패     | 매핑을 해제하고 대체 경로 사용            |
| 피어 액세스가 이미 활성화됨    | 계속 진행                                  |
| 피어 액세스가 이미 비활성화됨   | 계속 진행                                  |

cuda-oxide 통합 테스트는 컴퓨터에 GPU가 두 개 미만이거나 선택한 토폴로지에서 P2P를 지원하지 않으면 조기에 반환합니다. 따라서 단일 GPU 개발 시스템에서도 테스트 모음을 실행할 수 있으며, 호환되는 멀티 GPU 호스트에서는 해당 기능을 검증할 수 있습니다.

## API 요약

### VMM

| API                                        | 설명                                  |
| :----------------------------------------- | :------------------------------------------- |
| `vmm::allocation_granularity(device)`      | VMM 작업에 필요한 크기 단위 |
| `vmm::align_size(size, granularity)`       | 요청 크기를 올림 정렬                |
| `PhysicalAllocation::new(device, size)`    | 디바이스에 물리 메모리 할당         |
| `PhysicalAllocation::handle()`             | 원시 할당 핸들 반환             |
| `PhysicalAllocation::size()`               | 물리 할당 크기 반환          |
| `VirtualReservation::new(size, alignment)` | 가상 주소 범위 예약              |
| `VirtualReservation::base()`               | 예약된 기본 주소 반환             |
| `VirtualReservation::size()`               | 예약된 범위 크기 반환               |
| `Mapping::new(va, size, physical, offset)` | 가상 범위에 물리 메모리 매핑     |
| `Mapping::va()`                            | 매핑된 가상 주소 반환            |
| `Mapping::size()`                          | 매핑된 크기 반환                       |
| `vmm::set_access(va, size, devices)`       | 선택한 디바이스에 읽기/쓰기 권한 부여  |

### P2P

| API                                   | 설명                     |
| :------------------------------------ | :------------------------------ |
| `peer::can_access_peer(from, to)`     | 토폴로지 지원 여부 조회          |
| `peer::enable_peer_access(from, to)`  | 방향별 피어 액세스 활성화  |
| `peer::disable_peer_access(from, to)` | 방향별 피어 액세스 비활성화 |

## 다음 단계

실제 운영용 멀티 GPU 코드에서는 P2P를 지원하지 않는 시스템을 위한 대체 경로를 명시적으로 추가하세요. 일반적인 대안은 모든 멀티 GPU 토폴로지에서 직접 접근이 가능하다고 가정하지 않고, 페이지 잠금 호스트 메모리를 통해 데이터를 복사하는 것입니다.

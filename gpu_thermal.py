"""Read GPU die sensors from AppleSMC on Apple Silicon Macs."""

import ctypes as C
import platform
import struct
import subprocess


DEFAULT_GPU_LIMIT = 98.0
POLL_SECONDS = 2.0


class _KeyInfo(C.Structure):
    _fields_ = [("size", C.c_uint32), ("type", C.c_uint32), ("attributes", C.c_uint8)]


class _SMCRequest(C.Structure):
    # The version and power-limit fields are reserved for these read commands.
    _fields_ = [("key", C.c_uint32), ("reserved", C.c_uint8 * 24), ("info", _KeyInfo),
                ("result", C.c_uint8), ("status", C.c_uint8), ("command", C.c_uint8),
                ("index", C.c_uint32), ("data", C.c_uint8 * 32)]


class AppleSMCGPU:
    def __init__(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise OSError("需要 Apple Silicon Mac 的 GPU 温度传感器")
        self.io = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        for name, args, result in (
            ("IOServiceNameMatching", [C.c_char_p], C.c_void_p),
            ("IOServiceGetMatchingService", [C.c_uint32, C.c_void_p], C.c_uint32),
            ("IOServiceOpen", [C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint32)], C.c_int),
            ("IOObjectRelease", [C.c_uint32], C.c_int),
            ("IOServiceClose", [C.c_uint32], C.c_int),
            ("IOConnectCallStructMethod", [C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t,
                                          C.c_void_p, C.POINTER(C.c_size_t)], C.c_int),
        ):
            function = getattr(self.io, name)
            function.argtypes, function.restype = args, result
        task = C.c_uint32.in_dll(C.CDLL("/usr/lib/libSystem.B.dylib"), "mach_task_self_").value
        service = self.io.IOServiceGetMatchingService(0, self.io.IOServiceNameMatching(b"AppleSMCKeysEndpoint"))
        if not service:
            raise OSError("找不到 AppleSMC 温度传感器")
        connection = C.c_uint32()
        try:
            if self.io.IOServiceOpen(service, task, 0, C.byref(connection)):
                raise OSError("无法打开 AppleSMC 温度传感器")
        finally:
            self.io.IOObjectRelease(service)
        self.connection = connection.value
        try:
            self.sensors = self._discover()
        except BaseException:
            self.close()
            raise

    def _call(self, request: _SMCRequest) -> _SMCRequest | None:
        response = _SMCRequest()
        size = C.c_size_t(C.sizeof(response))
        status = self.io.IOConnectCallStructMethod(self.connection, 2, C.byref(request), size.value,
                                                  C.byref(response), C.byref(size))
        return None if status or response.result else response

    def _read(self, key: int, info: _KeyInfo) -> bytes:
        response = self._call(_SMCRequest(key=key, info=info, command=5))
        return bytes(response.data[:info.size]) if response is not None else b""

    def _discover(self) -> list[tuple[int, _KeyInfo]]:
        count_key = int.from_bytes(b"#KEY", "big")
        info = self._call(_SMCRequest(key=count_key, command=9))
        count = int.from_bytes(self._read(count_key, info.info), "big") if info is not None else 0
        try:
            chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                           stderr=subprocess.DEVNULL, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            chip = ""
        prefix = b"Tf" if "Apple M3" in chip else b"Tg"
        sensors = []
        for index in range(min(count, 16384)):
            entry = self._call(_SMCRequest(command=8, index=index))
            if entry is None or not entry.key.to_bytes(4, "big").startswith(prefix):
                continue
            info = self._call(_SMCRequest(key=entry.key, command=9))
            if info is not None and info.info.size == 4 and info.info.type == int.from_bytes(b"flt ", "big"):
                sensors.append((entry.key, info.info))
        return sensors

    def read(self) -> float | None:
        values = []
        for key, info in self.sensors:
            data = self._read(key, info)
            if data:
                temperature = struct.unpack("<f", data)[0]
                if 0 < temperature <= 150:
                    values.append(temperature)
        return max(values, default=None)

    def close(self) -> None:
        self.io.IOServiceClose(self.connection)

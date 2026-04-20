"""
Standalone OIDN diagnostic — bypasses sapien completely.

What this script does:
    1. Locate the OIDN shared libraries bundled with sapien.
    2. Load them via ctypes and call OIDN's C API directly.
    3. Print the real OIDN version that is loaded at runtime.
    4. Enumerate OIDN devices (CPU / CUDA).
    5. Create a tiny 64x64 color buffer, run the "RT" denoise filter once,
       and report whether the kernel succeeded.

Why:
    If this script crashes / fails with 'illegal memory access', the bug is
    100% in the OIDN-vs-driver stack (nothing to do with sapien).
    If this script succeeds but sapien still fails, the bug is in how sapien
    drives OIDN (e.g. Vulkan<->CUDA interop).

Usage (on the Linux server):
    # Default: auto-locate OIDN libs inside the active sapien install.
    python script/test_oidn_standalone.py

    # Or explicitly point to a directory that contains libOpenImageDenoise*.so*
    python script/test_oidn_standalone.py --oidn-dir /path/to/oidn/lib

    # Or pick a specific device (0 = default, usually CPU; try 1 for CUDA)
    python script/test_oidn_standalone.py --device-id 0
"""

import argparse
import ctypes
import ctypes.util
import glob
import os
import sys
import traceback


# OIDN C API constants (from OpenImageDenoise/oidn.h, ABI-stable across 2.x)
OIDN_DEVICE_TYPE_DEFAULT = 0
OIDN_DEVICE_TYPE_CPU = 1
OIDN_DEVICE_TYPE_SYCL = 2
OIDN_DEVICE_TYPE_CUDA = 3
OIDN_DEVICE_TYPE_HIP = 4
OIDN_DEVICE_TYPE_METAL = 5

OIDN_FORMAT_FLOAT3 = 3   # 3x float32, packed

DEVICE_TYPE_NAMES = {
    OIDN_DEVICE_TYPE_DEFAULT: "DEFAULT",
    OIDN_DEVICE_TYPE_CPU: "CPU",
    OIDN_DEVICE_TYPE_SYCL: "SYCL",
    OIDN_DEVICE_TYPE_CUDA: "CUDA",
    OIDN_DEVICE_TYPE_HIP: "HIP",
    OIDN_DEVICE_TYPE_METAL: "METAL",
}


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64, flush=True)


# ---------------------------------------------------------------------------
# Locate OIDN libs (cross-platform)
# ---------------------------------------------------------------------------

IS_WINDOWS = sys.platform.startswith("win")


def _default_oidn_dir() -> str:
    """Find sapien's bundled OIDN directory.

    - Linux : <sapien>/oidn_library/libOpenImageDenoise*.so*
    - Windows: <sapien>/OpenImageDenoise*.dll   (flat in sapien package root)
    """
    try:
        import sapien

        sapien_root = os.path.dirname(sapien.__file__)
    except Exception:
        return ""

    candidates = []
    if IS_WINDOWS:
        candidates.append(sapien_root)
    else:
        candidates.append(os.path.join(sapien_root, "oidn_library"))

    for d in candidates:
        if os.path.isdir(d):
            # Require at least one OIDN file to be present
            if glob.glob(os.path.join(d, "*OpenImageDenoise*")):
                return d
    return ""


def _find_lib(oidn_dir: str, basename: str) -> str:
    """Find the loadable OIDN library for `basename` inside `oidn_dir`.

    `basename` must be the Linux-style prefix WITHOUT extension, e.g.
    'libOpenImageDenoise_core'. On Windows the `lib` prefix is stripped
    automatically.
    """
    if IS_WINDOWS:
        # Strip 'lib' prefix on Windows: libOpenImageDenoise_core -> OpenImageDenoise_core
        win_base = basename[3:] if basename.startswith("lib") else basename
        patterns = [
            os.path.join(oidn_dir, f"{win_base}.dll"),
        ]
    else:
        patterns = [
            os.path.join(oidn_dir, f"{basename}.so"),
            os.path.join(oidn_dir, f"{basename}.so.*"),
        ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    # Prefer files with a version suffix (e.g. .so.2.3.0 over .so)
    candidates.sort(key=lambda s: (len(s), s), reverse=True)
    for c in candidates:
        if os.path.isfile(c) or os.path.islink(c):
            return c
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oidn-dir",
        default="",
        help="Directory containing libOpenImageDenoise*.so*. "
             "Defaults to sapien's bundled oidn_library/ dir.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="Which OIDN physical device to use. -1 = iterate and try each.",
    )
    parser.add_argument(
        "--width", type=int, default=64,
        help="Probe image width (pixels).",
    )
    parser.add_argument(
        "--height", type=int, default=64,
        help="Probe image height (pixels).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: locate libs
    # -----------------------------------------------------------------------
    section("[1] locate OIDN libraries")
    oidn_dir = args.oidn_dir or _default_oidn_dir()
    if not oidn_dir:
        print("ERR: cannot auto-detect OIDN directory. "
              "Pass --oidn-dir /path/to/oidn/lib")
        return 1
    print(f"oidn_dir = {oidn_dir}")

    core_lib_path = _find_lib(oidn_dir, "libOpenImageDenoise_core")
    main_lib_path = _find_lib(oidn_dir, "libOpenImageDenoise")
    cuda_lib_path = _find_lib(oidn_dir, "libOpenImageDenoise_device_cuda")

    print(f"core : {core_lib_path or '(not found)'}")
    print(f"main : {main_lib_path or '(not found)'}")
    print(f"cuda : {cuda_lib_path or '(not found)'}")

    if not core_lib_path or not main_lib_path:
        print("ERR: required OIDN libraries missing")
        return 1

    # -----------------------------------------------------------------------
    # Step 2: load libs
    # -----------------------------------------------------------------------
    section("[2] dlopen OIDN libraries")
    try:
        if IS_WINDOWS:
            # On Windows we must register the OIDN directory so that when
            # OpenImageDenoise.dll is loaded it can find its sibling core DLL.
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(oidn_dir)
            ctypes.CDLL(core_lib_path)
            print(f"  loaded: {core_lib_path}")
            oidn = ctypes.CDLL(main_lib_path)
            print(f"  loaded: {main_lib_path}")
        else:
            # On Linux load core first with RTLD_GLOBAL so main can see its
            # symbols, then load main.
            ctypes.CDLL(core_lib_path, ctypes.RTLD_GLOBAL)
            print(f"  loaded: {core_lib_path}")
            oidn = ctypes.CDLL(main_lib_path, ctypes.RTLD_GLOBAL)
            print(f"  loaded: {main_lib_path}")
    except OSError:
        print("ERR: failed to dlopen OIDN")
        traceback.print_exc()
        return 1

    # -----------------------------------------------------------------------
    # Step 3: bind C API prototypes we need
    # -----------------------------------------------------------------------
    section("[3] bind C API")

    # oidnGetNumPhysicalDevices() -> int
    oidn.oidnGetNumPhysicalDevices.restype = ctypes.c_int
    oidn.oidnGetNumPhysicalDevices.argtypes = []

    # oidnGetPhysicalDeviceInt(int id, const char* name) -> int
    oidn.oidnGetPhysicalDeviceInt.restype = ctypes.c_int
    oidn.oidnGetPhysicalDeviceInt.argtypes = [ctypes.c_int, ctypes.c_char_p]

    # oidnGetPhysicalDeviceString(int id, const char* name) -> const char*
    oidn.oidnGetPhysicalDeviceString.restype = ctypes.c_char_p
    oidn.oidnGetPhysicalDeviceString.argtypes = [ctypes.c_int, ctypes.c_char_p]

    # OIDNDevice oidnNewDeviceByID(int id)  (available since 2.1)
    has_new_by_id = hasattr(oidn, "oidnNewDeviceByID")
    if has_new_by_id:
        oidn.oidnNewDeviceByID.restype = ctypes.c_void_p
        oidn.oidnNewDeviceByID.argtypes = [ctypes.c_int]

    # OIDNDevice oidnNewDevice(OIDNDeviceType t)
    oidn.oidnNewDevice.restype = ctypes.c_void_p
    oidn.oidnNewDevice.argtypes = [ctypes.c_int]

    # void oidnCommitDevice(OIDNDevice)
    oidn.oidnCommitDevice.restype = None
    oidn.oidnCommitDevice.argtypes = [ctypes.c_void_p]

    # void oidnReleaseDevice(OIDNDevice)
    oidn.oidnReleaseDevice.restype = None
    oidn.oidnReleaseDevice.argtypes = [ctypes.c_void_p]

    # OIDNError oidnGetDeviceError(OIDNDevice, const char** outMessage)
    oidn.oidnGetDeviceError.restype = ctypes.c_int
    oidn.oidnGetDeviceError.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]

    # OIDNBuffer oidnNewBuffer(OIDNDevice, size_t byteSize)
    oidn.oidnNewBuffer.restype = ctypes.c_void_p
    oidn.oidnNewBuffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    # void* oidnGetBufferData(OIDNBuffer)
    oidn.oidnGetBufferData.restype = ctypes.c_void_p
    oidn.oidnGetBufferData.argtypes = [ctypes.c_void_p]

    # void oidnReleaseBuffer(OIDNBuffer)
    oidn.oidnReleaseBuffer.restype = None
    oidn.oidnReleaseBuffer.argtypes = [ctypes.c_void_p]

    # OIDNFilter oidnNewFilter(OIDNDevice, const char* type)
    oidn.oidnNewFilter.restype = ctypes.c_void_p
    oidn.oidnNewFilter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # void oidnSetFilterImage(filter, name, buffer, format, w, h,
    #                         byteOffset, pixelByteStride, rowByteStride)
    oidn.oidnSetFilterImage.restype = None
    oidn.oidnSetFilterImage.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]

    # void oidnSetFilterBool(filter, name, bool)
    oidn.oidnSetFilterBool.restype = None
    oidn.oidnSetFilterBool.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]

    # void oidnCommitFilter(filter)
    oidn.oidnCommitFilter.restype = None
    oidn.oidnCommitFilter.argtypes = [ctypes.c_void_p]

    # void oidnExecuteFilter(filter)   <-- this is where illegal mem access fires
    oidn.oidnExecuteFilter.restype = None
    oidn.oidnExecuteFilter.argtypes = [ctypes.c_void_p]

    # void oidnReleaseFilter(filter)
    oidn.oidnReleaseFilter.restype = None
    oidn.oidnReleaseFilter.argtypes = [ctypes.c_void_p]

    # oidnGetDeviceInt for version
    oidn.oidnGetDeviceInt.restype = ctypes.c_int
    oidn.oidnGetDeviceInt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    print("  all symbols bound")

    # -----------------------------------------------------------------------
    # Step 4: enumerate physical devices and print versions
    # -----------------------------------------------------------------------
    section("[4] enumerate OIDN physical devices")
    n_dev = oidn.oidnGetNumPhysicalDevices()
    print(f"  number of physical devices : {n_dev}")

    device_infos = []
    for i in range(n_dev):
        try:
            name = oidn.oidnGetPhysicalDeviceString(i, b"name") or b""
            dtype_i = oidn.oidnGetPhysicalDeviceInt(i, b"type")
            ver_major = oidn.oidnGetPhysicalDeviceInt(i, b"systemMemorySupported")
            uuid = oidn.oidnGetPhysicalDeviceString(i, b"uuid") or b""
            print(f"  device[{i}] type={DEVICE_TYPE_NAMES.get(dtype_i, dtype_i)}"
                  f" name={name.decode('utf-8', errors='replace')!r}")
            device_infos.append((i, dtype_i))
        except Exception:
            print(f"  device[{i}] (error querying properties)")
            traceback.print_exc()
            device_infos.append((i, -1))

    # -----------------------------------------------------------------------
    # Step 5: run the denoise probe on each device (or a specific one)
    # -----------------------------------------------------------------------
    section("[5] run denoise probe")

    if args.device_id >= 0:
        targets = [args.device_id]
    else:
        # Try every device. If CUDA is present we really want to know whether
        # it works, because that is the path sapien uses.
        targets = [info[0] for info in device_infos] if device_infos else [-1]

    overall_ok = False
    for dev_id in targets:
        print(f"\n-- probing device_id = {dev_id} --", flush=True)
        ok = _probe_one_device(oidn, dev_id, args.width, args.height,
                               has_new_by_id=has_new_by_id)
        print(f"   result: {'OK' if ok else 'FAIL'}")
        overall_ok = overall_ok or ok

    section("SUMMARY")
    for dev_id, dtype in device_infos:
        print(f"  device[{dev_id}] type={DEVICE_TYPE_NAMES.get(dtype, dtype)}")
    print(f"  overall denoise OK: {overall_ok}")
    return 0 if overall_ok else 2


def _probe_one_device(oidn, device_id: int, width: int, height: int,
                      has_new_by_id: bool) -> bool:
    """Create device -> buffer -> filter -> execute. Return True on success."""
    # Step 1: create device
    if device_id < 0 or not has_new_by_id:
        print("   oidnNewDevice(DEFAULT) ...", flush=True)
        dev = oidn.oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT)
    else:
        print(f"   oidnNewDeviceByID({device_id}) ...", flush=True)
        dev = oidn.oidnNewDeviceByID(device_id)
    if not dev:
        print("   ERR: device creation returned null")
        return False

    try:
        # Commit device
        print("   oidnCommitDevice(dev) ...", flush=True)
        oidn.oidnCommitDevice(dev)
        err_msg = ctypes.c_char_p()
        err = oidn.oidnGetDeviceError(dev, ctypes.byref(err_msg))
        if err != 0:
            msg = err_msg.value.decode("utf-8") if err_msg.value else ""
            print(f"   ERR: oidnGetDeviceError={err} msg={msg!r}")
            return False

        # Print runtime version once we have a device
        try:
            v_major = oidn.oidnGetDeviceInt(dev, b"versionMajor")
            v_minor = oidn.oidnGetDeviceInt(dev, b"versionMinor")
            v_patch = oidn.oidnGetDeviceInt(dev, b"versionPatch")
            print(f"   OIDN version in use : {v_major}.{v_minor}.{v_patch}",
                  flush=True)
        except Exception:
            pass

        # Step 2: allocate color/output buffers on this device
        nbytes = width * height * 3 * 4  # float3 per pixel
        print(f"   oidnNewBuffer(color, {nbytes} bytes) ...", flush=True)
        buf_color = oidn.oidnNewBuffer(dev, nbytes)
        buf_output = oidn.oidnNewBuffer(dev, nbytes)
        if not buf_color or not buf_output:
            print("   ERR: buffer allocation failed")
            return False

        # Fill the color buffer with a simple gradient so the denoiser has
        # something reasonable to work on.
        ptr = oidn.oidnGetBufferData(buf_color)
        if ptr:
            # Write via ctypes
            fill = (ctypes.c_float * (width * height * 3))()
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * 3
                    fill[idx + 0] = x / max(1, width - 1)
                    fill[idx + 1] = y / max(1, height - 1)
                    fill[idx + 2] = 0.5
            ctypes.memmove(ptr, fill, nbytes)

        # Step 3: build filter
        print("   oidnNewFilter('RT') ...", flush=True)
        flt = oidn.oidnNewFilter(dev, b"RT")
        if not flt:
            print("   ERR: filter creation failed")
            return False

        oidn.oidnSetFilterImage(flt, b"color", buf_color,
                                OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0)
        oidn.oidnSetFilterImage(flt, b"output", buf_output,
                                OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0)
        oidn.oidnSetFilterBool(flt, b"hdr", True)

        print("   oidnCommitFilter(flt) ...", flush=True)
        oidn.oidnCommitFilter(flt)

        err = oidn.oidnGetDeviceError(dev, ctypes.byref(err_msg))
        if err != 0:
            msg = err_msg.value.decode("utf-8") if err_msg.value else ""
            print(f"   ERR after CommitFilter: code={err} msg={msg!r}")
            return False

        # Step 4: THE KERNEL CALL — this is where sapien blows up
        print("   oidnExecuteFilter(flt)  <-- denoiser kernel fires here",
              flush=True)
        oidn.oidnExecuteFilter(flt)

        err = oidn.oidnGetDeviceError(dev, ctypes.byref(err_msg))
        if err != 0:
            msg = err_msg.value.decode("utf-8") if err_msg.value else ""
            print(f"   ERR after ExecuteFilter: code={err} msg={msg!r}")
            return False

        print("   denoise kernel completed without OIDN error")
        return True
    finally:
        try:
            oidn.oidnReleaseDevice(dev)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

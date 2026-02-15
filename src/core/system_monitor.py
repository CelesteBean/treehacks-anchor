"""Real-time system metrics for Jetson Orin Nano — CPU, RAM, GPU, power, pipeline latency.

Collects metrics via psutil, tegrastats (Jetson-specific), and /sys paths.
Publishes to ZMQ port 5559 every 1 second for dashboard display.

Usage::
    python -m src.core.system_monitor
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import zmq

from src.core.message_bus import (
    MessageBus,
    SYSTEM_PORT,
    TACTIC_PORT,
    TRANSCRIPT_PORT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process name patterns (matched against cmdline)
# ---------------------------------------------------------------------------

PIPELINE_PROCESS_PATTERNS: dict[str, str] = {
    "audio_capture": "audio_capture",
    "speech_recognition": "speech_recognition",
    "content_analyzer": "content_analyzer",
    "audio_intervention": "audio_intervention",
    "judges_window": "judges_window",
}


@dataclass
class PipelineLatency:
    """Rolling averages of inference latency (ms) per pipeline stage."""

    whisper: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    analyzer: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    tts: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def add_whisper(self, ms: float) -> None:
        self.whisper.append(ms)

    def add_analyzer(self, ms: float) -> None:
        self.analyzer.append(ms)

    def add_tts(self, ms: float) -> None:
        self.tts.append(ms)

    def to_dict(self) -> dict[str, Any]:
        def _avg(d: deque) -> float | None:
            return round(sum(d) / len(d), 1) if d else None

        return {
            "whisper_ms": _avg(self.whisper),
            "analyzer_ms": _avg(self.analyzer),
            "tts_ms": _avg(self.tts),
            "whisper_samples": len(self.whisper),
            "analyzer_samples": len(self.analyzer),
            "tts_samples": len(self.tts),
        }


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


def _read_file(path: str, default: float = 0.0) -> float:
    """Read numeric value from sysfs file."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return default


def _find_hwmon_power() -> str | None:
    """Find first in*_input under ina3221 hwmon (power in milliwatts)."""
    base = "/sys/bus/i2c/drivers/ina3221"
    if not os.path.exists(base):
        base = "/sys/bus/i2c/drivers/ina3221x"
    if not os.path.exists(base):
        return None
    for dev in glob.glob(f"{base}/*/hwmon/hwmon*/in*_input"):
        if "power" in os.path.basename(dev) or dev.endswith("_input"):
            return dev
    for dev in glob.glob(f"{base}/*/iio:device*/in_power0_input"):
        return dev
    return None


def _parse_tegrastats_line(line: str) -> dict[str, Any]:
    """Parse one tegrastats output line into structured metrics."""
    out: dict[str, Any] = {}

    # RAM: "RAM 2345/7620MB (lfb 1x2MB) SWAP 0/3810MB (cached 0MB)"
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if ram_match:
        out["ram_used_mb"] = int(ram_match.group(1))
        out["ram_total_mb"] = int(ram_match.group(2))
    swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
    if swap_match:
        out["swap_used_mb"] = int(swap_match.group(1))
        out["swap_total_mb"] = int(swap_match.group(2))

    # CPU: "CPU [12%@1510,10%@1510,8%@1510,15%@1510,0%@1510,0%@1510]"
    cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
    if cpu_match:
        parts = cpu_match.group(1).split(",")
        per_core: list[float] = []
        freq_mhz: float | None = None
        for p in parts:
            m = re.match(r"(\d+)%@(\d+)", p.strip())
            if m:
                per_core.append(float(m.group(1)))
                if freq_mhz is None:
                    freq_mhz = float(m.group(2))
        out["cpu_percent_per_core"] = per_core
        out["cpu_freq_mhz"] = freq_mhz

    # EMC: "EMC_FREQ 0%@2133"
    emc_match = re.search(r"EMC_FREQ\s+\d+%@(\d+)", line)
    if emc_match:
        out["emc_freq_mhz"] = int(emc_match.group(1))

    # GPU: "GR3D_FREQ 18%" or "GR3D_FREQ 12%@1300" (Jetson Orin)
    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if gpu_match:
        out["gpu_percent"] = int(gpu_match.group(1))

    # GPU temperature: "gpu@47.125C"
    gpu_temp_match = re.search(r"gpu@([\d.]+)C", line)
    if gpu_temp_match:
        out["gpu_temp_c"] = round(float(gpu_temp_match.group(1)), 1)

    # GPU frequency (optional): "GR3D_FREQ 12%@1300"
    gpu_freq_match = re.search(r"GR3D_FREQ\s+\d+%@\[?(\d+)", line)
    if gpu_freq_match:
        out["gpu_freq_mhz"] = int(gpu_freq_match.group(1))

    # Power: "VDD_CPU_GPU_CV 8210/8045" or "VDD_IN 8234"
    for pat in [r"VDD_CPU_GPU_CV\s+(\d+)", r"VDD_IN\s+(\d+)", r"POM_5V_IN\s+(\d+)"]:
        pow_match = re.search(pat, line)
        if pow_match:
            out["power_mw"] = int(pow_match.group(1))
            break

    return out


def _try_tegrastats(interval_ms: int = 1000) -> dict[str, Any] | None:
    """Run tegrastats once and parse output. Returns None if unavailable."""
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", str(interval_ms)],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        lines = proc.stdout.strip().split("\n")
        return _parse_tegrastats_line(lines[-1]) if lines else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_per_process_memory() -> dict[str, float]:
    """Get RSS in MB for pipeline processes by matching cmdline."""
    try:
        import psutil
    except ImportError:
        return {k: 0.0 for k in PIPELINE_PROCESS_PATTERNS}

    result: dict[str, float] = {k: 0.0 for k in PIPELINE_PROCESS_PATTERNS}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmdline) if isinstance(cmdline, (list, tuple)) else str(cmdline)
            mem = proc.info.get("memory_info")
            rss_mb = (mem.rss / (1024 * 1024)) if mem else 0.0

            for key, pattern in PIPELINE_PROCESS_PATTERNS.items():
                if pattern in cmd_str:
                    result[key] = result[key] + rss_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


def _get_cpu_psutil() -> dict[str, Any]:
    """CPU metrics via psutil."""
    try:
        import psutil
    except ImportError:
        return {"percent": 0, "percent_per_core": [], "freq_mhz": None}

    perc = psutil.cpu_percent(interval=None)
    perc_per_core = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()
    return {
        "percent": round(perc, 1),
        "percent_per_core": [round(x, 1) for x in perc_per_core],
        "freq_mhz": round(freq.current, 1) if freq else None,
    }


def _get_memory_psutil() -> dict[str, Any]:
    """Memory metrics via psutil."""
    try:
        import psutil
    except ImportError:
        return {"total_mb": 0, "used_mb": 0, "available_mb": 0, "percent": 0}

    v = psutil.virtual_memory()
    return {
        "total_mb": round(v.total / (1024 * 1024), 1),
        "used_mb": round(v.used / (1024 * 1024), 1),
        "available_mb": round(v.available / (1024 * 1024), 1),
        "percent": round(v.percent, 1),
    }


def _get_cpu_temp() -> float | None:
    """CPU temperature in Celsius (thermal_zone or hwmon)."""
    for path in [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
    ]:
        if os.path.exists(path):
            val = _read_file(path)
            return round(val / 1000.0, 1) if val > 0 else None
    return None


def _get_gpu_temp() -> float | None:
    """GPU temperature (Jetson hwmon)."""
    for p in glob.glob("/sys/devices/gpu.0/*/temp1_input"):
        val = _read_file(p)
        if val > 0:
            return round(val / 1000.0, 1)
    return None


def _get_power_sysfs() -> float | None:
    """Power in milliwatts from INA3221 hwmon."""
    path = _find_hwmon_power()
    if path:
        return _read_file(path)
    return None


def _get_power_mode() -> str:
    """Infer power mode from nvpmodel or sysfs (15W, 7W, etc.)."""
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout:
            if "15W" in out.stdout or "MAXN" in out.stdout:
                return "15W"
            if "7W" in out.stdout or "10W" in out.stdout:
                return out.stdout.split()[0] if out.stdout.split() else "7W"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "—"


# ---------------------------------------------------------------------------
# SystemMonitor
# ---------------------------------------------------------------------------


class SystemMonitor:
    """Collect system metrics and publish to ZMQ every interval_sec."""

    def __init__(
        self,
        bus: MessageBus | None = None,
        port: int = SYSTEM_PORT,
        interval_sec: float = 1.0,
    ) -> None:
        self.bus = bus or MessageBus()
        self.port = port
        self.interval_sec = interval_sec
        self._latency = PipelineLatency()
        self._stop = threading.Event()
        self._tegrastats_proc: subprocess.Popen | None = None

    def _collect_metrics(self) -> dict[str, Any]:
        """Gather all metrics from available sources."""
        ts = datetime.now(timezone.utc).isoformat()

        # Base metrics from psutil
        cpu = _get_cpu_psutil()
        mem = _get_memory_psutil()
        mem["per_process_mb"] = _get_per_process_memory()
        mem["swap"] = _get_swap_psutil()

        # Temperatures
        cpu_temp = _get_cpu_temp()
        gpu_temp = _get_gpu_temp()

        # Jetson-specific: tegrastats
        tegra = _try_tegrastats(1000)
        if tegra:
            if "ram_used_mb" in tegra:
                mem["used_mb"] = tegra["ram_used_mb"]
                mem["total_mb"] = tegra["ram_total_mb"]
                mem["percent"] = round(100.0 * tegra["ram_used_mb"] / tegra["ram_total_mb"], 1)
            if "cpu_percent_per_core" in tegra:
                cpu["percent_per_core"] = tegra["cpu_percent_per_core"]
                cpu["percent"] = round(
                    sum(tegra["cpu_percent_per_core"]) / len(tegra["cpu_percent_per_core"]), 1
                )
            if "cpu_freq_mhz" in tegra:
                cpu["freq_mhz"] = tegra["cpu_freq_mhz"]
            if "gpu_percent" in tegra:
                pass  # used in gpu block below
            if "gpu_freq_mhz" in tegra:
                pass  # used in gpu block below

        # GPU block (Jetson tegrastats / fallback)
        gpu: dict[str, Any] = {
            "percent": tegra.get("gpu_percent") if tegra else None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "freq_mhz": tegra.get("gpu_freq_mhz") if tegra else None,
            "temp_c": tegra.get("gpu_temp_c") if tegra else gpu_temp,
        }
        gpu_mem = _get_gpu_memory()
        if gpu_mem:
            gpu["memory_used_mb"] = gpu_mem.get("used_mb")
            gpu["memory_total_mb"] = gpu_mem.get("total_mb")

        # Power
        power_mw = tegra.get("power_mw") if tegra else _get_power_sysfs()
        power: dict[str, Any] = {
            "current_mw": power_mw,
            "current_w": round(power_mw / 1000.0, 2) if power_mw else None,
            "mode": _get_power_mode(),
        }

        # Pipeline latency
        pipeline = self._latency.to_dict()
        pipeline["e2e_ms"] = None
        w, a, t = pipeline.get("whisper_ms"), pipeline.get("analyzer_ms"), pipeline.get("tts_ms")
        if w is not None or a is not None or t is not None:
            pipeline["e2e_ms"] = round((w or 0) + (a or 0) + (t or 0), 1)

        return {
            "timestamp": ts,
            "cpu": {**cpu, "temp_c": cpu_temp},
            "memory": mem,
            "gpu": gpu,
            "power": power,
            "pipeline": pipeline,
        }

    def _zmq_listener(self) -> None:
        """Subscribe to transcript and tactic ports to capture latency."""
        try:
            sub = self.bus.create_subscriber(
                ports=[TRANSCRIPT_PORT, TACTIC_PORT],
                topics=["transcript", "tactics"],
            )
            time.sleep(0.3)
            while not self._stop.is_set():
                result = self.bus.receive(sub, timeout_ms=200)
                if result is None:
                    continue
                topic, envelope = result
                data = envelope.get("data", {})
                if topic == "transcript":
                    lat = data.get("inference_time_ms")
                    if lat is not None:
                        self._latency.add_whisper(float(lat))
                elif topic == "tactics":
                    lat = data.get("inference_time_ms")
                    if lat is not None:
                        self._latency.add_analyzer(float(lat))
        except Exception:
            logger.exception("ZMQ listener error")

    def _run_publisher(self) -> None:
        """Publish metrics every interval_sec."""
        pub = self.bus.create_publisher(self.port)
        logger.info("System monitor publishing on port %d every %.1fs", self.port, self.interval_sec)
        while not self._stop.is_set():
            try:
                metrics = self._collect_metrics()
                self.bus.publish(pub, "system", metrics)
            except Exception:
                logger.exception("Error collecting metrics")
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        """Start listener thread and publisher loop (blocking)."""
        listener = threading.Thread(target=self._zmq_listener, daemon=True)
        listener.start()
        try:
            self._run_publisher()
        finally:
            self._stop.set()


def _get_swap_psutil() -> dict[str, Any]:
    """Swap usage via psutil."""
    try:
        import psutil
    except ImportError:
        return {"used_mb": 0, "total_mb": 0, "percent": 0}
    s = psutil.swap_memory()
    return {
        "used_mb": round(s.used / (1024 * 1024), 1),
        "total_mb": round(s.total / (1024 * 1024), 1),
        "percent": round(s.percent, 1) if s.total > 0 else 0,
    }


def _get_gpu_memory() -> dict[str, float] | None:
    """GPU memory used/total in MB (nvidia-smi or tegrastats)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout:
            parts = out.stdout.strip().split(",")
            if len(parts) >= 2:
                used_s = parts[0].strip().replace("[N/A]", "0").split()[0]
                total_s = parts[1].strip().replace("[N/A]", "0").split()[0]
                try:
                    used = float(used_s)
                    total = float(total_s)
                    if total > 0 or used > 0:
                        return {"used_mb": used, "total_mb": total}
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    monitor = SystemMonitor()
    try:
        monitor.start()
    except KeyboardInterrupt:
        pass

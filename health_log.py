"""
health_log.py
보드 상태(온도·부하·메모리·CPU 클록)를 logs/health.log 에 10초마다 남긴다. 줄마다 디스크에 확정한다.

2026-11-14 보드가 하루에 여러 번 갑자기 재부팅됐는데, 루트 파일시스템이 commit=600 이라
꺼지기 직전 기록(syslog·sysstat)이 모두 사라져 원인을 알 수 없었다. 다음에는 직전 상태가 남도록 한다.
웹 서버가 켜질 때 함께 시작된다 (app.py).
"""
import glob
import os
import threading
import time

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "health.log")
EVERY_S = 10
MAX_BYTES = 2_000_000


def _read(p, default="?"):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def sample() -> str:
    temps = []
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = _read(z + "/temp")
        temps.append(f"{_read(z + '/type').split('-')[0]}={int(t) // 1000 if t.isdigit() else '?'}")
    mem = dict(line.split(":", 1) for line in _read("/proc/meminfo", "").splitlines() if ":" in line)
    avail = int(mem.get("MemAvailable", "0 kB").split()[0]) // 1024
    freqs = [_read(f"/sys/devices/system/cpu/cpufreq/policy{p}/scaling_cur_freq") for p in (0, 4, 6)]
    freqs = "/".join(str(int(f) // 1000) if f.isdigit() else "?" for f in freqs)
    load = _read("/proc/loadavg").split()[:3]
    return f"{time.strftime('%Y-%m-%d %H:%M:%S')} load={','.join(load)} avail_mb={avail} mhz={freqs} {' '.join(temps)}"


def _run():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    while True:
        try:
            if os.path.exists(PATH) and os.path.getsize(PATH) > MAX_BYTES:
                os.replace(PATH, PATH + ".1")
            with open(PATH, "a") as f:
                f.write(sample() + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
        time.sleep(EVERY_S)


def start():
    threading.Thread(target=_run, name="health-log", daemon=True).start()

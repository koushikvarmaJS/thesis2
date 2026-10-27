"""Jetson power/thermal/memory sampler.

Reads the onboard INA3221 power monitor through sysfs (hwmon), falling back to
parsing `tegrastats` if sysfs isn't available. Samples run in a background
thread for the whole benchmark; energy for any time window is computed
afterwards from the stored samples (trapezoid rule).

Run directly for a sanity check:  python3 power.py --watch 10
"""
import argparse
import glob
import os
import re
import subprocess
import threading
import time


def _read(path):
    with open(path) as f:
        return f.read().strip()


# ---------------------------------------------------------------- discovery

def discover_hwmon_rails():
    """INA3221 via hwmon (JetPack 6 / L4T 36): {label: (mV path, mA path)}."""
    rails = {}
    patterns = ['/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*',
                '/sys/class/hwmon/hwmon*']
    for pat in patterns:
        for hw in glob.glob(pat):
            try:
                if 'ina3221' not in _read(os.path.join(hw, 'name')).lower():
                    continue
            except OSError:
                continue
            for lab in glob.glob(os.path.join(hw, 'in*_label')):
                idx = re.search(r'in(\d+)_label', lab).group(1)
                v = os.path.join(hw, f'in{idx}_input')
                c = os.path.join(hw, f'curr{idx}_input')
                if os.path.exists(v) and os.path.exists(c):
                    rails.setdefault(_read(lab), (v, c))
        if rails:
            break
    return rails


def discover_iio_rails():
    """INA3221 via iio (JetPack 5 / L4T 35): {label: power path (mW)}."""
    rails = {}
    for dev in glob.glob('/sys/bus/i2c/drivers/ina3221x/*/iio:device*'):
        for name_f in glob.glob(os.path.join(dev, 'rail_name_*')):
            idx = name_f.rsplit('_', 1)[1]
            p = os.path.join(dev, f'in_power{idx}_input')
            if os.path.exists(p):
                rails[_read(name_f)] = p
    return rails


def discover_thermal():
    zones = {}
    for z in glob.glob('/sys/devices/virtual/thermal/thermal_zone*'):
        try:
            zones[_read(os.path.join(z, 'type'))] = os.path.join(z, 'temp')
        except OSError:
            pass
    return zones


def discover_gpu_freq():
    for pat in ['/sys/class/devfreq/*gpu*/cur_freq',
                '/sys/devices/platform/*gpu*/devfreq/*/cur_freq',
                '/sys/devices/platform/bus@0/*gpu*/devfreq/*/cur_freq']:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def discover_gpu_load():
    """GPU busy fraction file (0..1000) of the integrated GPU."""
    for pat in ['/sys/devices/platform/bus@0/*.gpu/load', '/sys/devices/platform/*gpu*/load',
                '/sys/devices/gpu.0/load']:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def cpu_jiffies():
    """(busy, total) jiffies summed over all CPUs, from /proc/stat."""
    with open('/proc/stat') as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
    return sum(v) - idle, sum(v)


def mem_available_mb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024.0
    return float('nan')


# ---------------------------------------------------------------- sampler

class PowerMeter:
    """Continuous background sampler.
    samples = [(t_monotonic, {rail: W}, mem_avail_mb, {'gpu_load': 0..1, 'cpu': (busy, total) jiffies})]."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._lock = threading.Lock()
        self._running = False
        self._proc = None
        self.hwmon = discover_hwmon_rails()
        self.iio = {} if self.hwmon else discover_iio_rails()
        self.thermal = discover_thermal()
        self.gpu_freq_path = discover_gpu_freq()
        self.gpu_load_path = discover_gpu_load()
        if self.hwmon:
            self.backend = 'sysfs-hwmon'
        elif self.iio:
            self.backend = 'sysfs-iio'
        else:
            self.backend = 'tegrastats'

    # -- backends
    def _read_sysfs(self):
        if self.hwmon:
            return {name: int(_read(v)) * int(_read(c)) / 1e6  # mV*mA -> W
                    for name, (v, c) in self.hwmon.items()}
        return {name: int(_read(p)) / 1000.0 for name, p in self.iio.items()}

    def _util(self):
        u = {'cpu': cpu_jiffies()}
        if self.gpu_load_path:
            try:
                u['gpu_load'] = int(_read(self.gpu_load_path)) / 1000.0
            except (OSError, ValueError):
                pass
        return u

    def _loop_sysfs(self):
        while self._running:
            try:
                w = self._read_sysfs()
                with self._lock:
                    self.samples.append((time.monotonic(), w, mem_available_mb(), self._util()))
            except OSError:
                pass
            time.sleep(self.interval)

    def _loop_tegrastats(self):
        ms = max(10, int(self.interval * 1000))
        self._proc = subprocess.Popen(['tegrastats', '--interval', str(ms)],
                                      stdout=subprocess.PIPE, text=True)
        rx = re.compile(r'(VDD_\w+|POM_\w+) (\d+)mW/(\d+)mW')
        for line in self._proc.stdout:
            if not self._running:
                break
            w = {m.group(1): int(m.group(2)) / 1000.0 for m in rx.finditer(line)}
            if w:
                with self._lock:
                    self.samples.append((time.monotonic(), w, mem_available_mb(), self._util()))

    def start(self):
        self._running = True
        target = self._loop_tegrastats if self.backend == 'tegrastats' else self._loop_sysfs
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        time.sleep(max(0.3, 3 * self.interval))  # let a few samples arrive
        return self

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
        self._thread.join(timeout=2)

    # -- queries
    def rails(self):
        with self._lock:
            return sorted(self.samples[-1][1]) if self.samples else []

    def window(self, t0, t1):
        with self._lock:
            return [s for s in self.samples if t0 <= s[0] <= t1]

    def energy(self, t0, t1, rail):
        """Joules on `rail` between monotonic times t0..t1 (trapezoid)."""
        s = self.window(t0, t1)
        if len(s) < 2:
            return float('nan'), len(s)
        e = 0.0
        for a, b in zip(s, s[1:]):
            e += 0.5 * (a[1].get(rail, 0) + b[1].get(rail, 0)) * (b[0] - a[0])
        # extend flat to the window edges so short windows aren't truncated
        e += s[0][1].get(rail, 0) * (s[0][0] - t0) + s[-1][1].get(rail, 0) * (t1 - s[-1][0])
        return e, len(s)

    def mean_power(self, t0, t1, rail):
        s = self.window(t0, t1)
        return sum(x[1].get(rail, 0) for x in s) / len(s) if s else float('nan')

    def min_mem_available(self, t0, t1):
        s = self.window(t0, t1)
        return min(x[2] for x in s) if s else float('nan')

    def gpu_load(self, t0, t1):
        """Mean GPU busy fraction (0..1) in the window, or None."""
        v = [x[3]['gpu_load'] for x in self.window(t0, t1) if 'gpu_load' in x[3]]
        return sum(v) / len(v) if v else None

    def cpu_busy(self, t0, t1):
        """Fraction of all CPU time that was busy in the window (0..1), from /proc/stat."""
        s = self.window(t0, t1)
        if len(s) < 2:
            return None
        (b0, a0), (b1, a1) = s[0][3]['cpu'], s[-1][3]['cpu']
        return (b1 - b0) / (a1 - a0) if a1 > a0 else None

    def temps(self):
        out = {}
        for name, p in self.thermal.items():
            try:
                out[name] = int(_read(p)) / 1000.0
            except (OSError, ValueError):
                pass
        return out

    def gpu_freq_mhz(self):
        try:
            return int(_read(self.gpu_freq_path)) / 1e6 if self.gpu_freq_path else None
        except (OSError, ValueError):
            return None


def main():
    ap = argparse.ArgumentParser(description='Live Jetson power readout (sanity check).')
    ap.add_argument('--watch', type=float, default=10, help='seconds to print readings')
    ap.add_argument('--interval', type=float, default=0.05)
    args = ap.parse_args()

    m = PowerMeter(args.interval).start()
    print(f'backend: {m.backend}')
    print(f'rails  : {m.rails()}')
    print(f'thermal: {sorted(m.thermal)}')
    print(f'gpu freq file: {m.gpu_freq_path}')
    t_start = time.monotonic()
    while time.monotonic() - t_start < args.watch:
        time.sleep(1)
        with m._lock:
            last = m.samples[-1] if m.samples else None
        if last:
            rails = '  '.join(f'{k}={v:6.2f}W' for k, v in sorted(last[1].items()))
            print(f'{rails}  mem_avail={last[2]:7.0f}MB  gpu={m.gpu_freq_mhz()}MHz')
    t_end = time.monotonic()
    n = len(m.window(t_start, t_end))
    print(f'\n{n} samples in {t_end - t_start:.1f}s  (~{n / (t_end - t_start):.0f} Hz)')
    for r in m.rails():
        e, _ = m.energy(t_start, t_end, r)
        print(f'{r:16s} mean {m.mean_power(t_start, t_end, r):6.2f} W   energy {e:7.1f} J')
    m.stop()


if __name__ == '__main__':
    main()

"""Extract only machine_hall/MH_01_easy/MH_01_easy.zip from the 12.7 GB remote
machine_hall.zip using remotezip (correct zip64 + HTTP range handling).
"""

from __future__ import annotations

import time
from pathlib import Path

from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main/machine_hall.zip"
TARGET = "machine_hall/MH_01_easy/MH_01_easy.zip"
OUT = Path(r"E:\KFPPS_data\euroc\MH_01_easy.zip")

OUT.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with RemoteZip(URL) as rz:
    info = rz.getinfo(TARGET)
    print(f"target {TARGET}: {info.file_size/1e6:.1f} MB (compress_type={info.compress_type})")
    with rz.open(TARGET) as src, OUT.open("wb") as dst:
        written = 0
        while True:
            chunk = src.read(16 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            written += len(chunk)
            mbps = written / 1e6 / max(time.time() - t0, 1e-6)
            print(f"  {written/1e6:7.1f} / {info.file_size/1e6:.1f} MB ({mbps:.1f} MB/s)", flush=True)
print(f"DONE -> {OUT} in {time.time()-t0:.0f}s")

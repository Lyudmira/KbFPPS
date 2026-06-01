"""Probe the remote machine_hall.zip central directory via HTTP range,
without downloading the whole 12 GB. Lists MH_01_easy entries and total size.
"""

from __future__ import annotations

import io
import struct
import sys
import urllib.request
import zipfile

URL = "https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main/machine_hall.zip"


def get_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def get_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


size = get_size(URL)
print(f"total zip size: {size/1e9:.2f} GB")

# Read the last 256 KB to capture EOCD (and zip64 EOCD if present).
tail_len = min(size, 262144)
tail = get_range(URL, size - tail_len, size - 1)

# Find EOCD signature 0x06054b50.
eocd = tail.rfind(b"PK\x05\x06")
if eocd < 0:
    sys.exit("no EOCD found")
# Parse classic EOCD.
# Parse classic EOCD (22 bytes): sig, disk, cd_disk, entries_disk, total_entries,
# cd_size, cd_offset, comment_len.
(_sig, _disk, _cd_disk, _ent_disk, total_entries, cd_size, cd_offset, _clen) = struct.unpack(
    "<IHHHHIIH", tail[eocd:eocd + 22]
)
print(f"classic EOCD: total_entries={total_entries} cd_size={cd_size} cd_offset={cd_offset}")

# Detect zip64 (values == 0xFFFFFFFF / 0xFFFF mean look at zip64 EOCD).
zip64 = cd_offset == 0xFFFFFFFF or total_entries == 0xFFFF or cd_size == 0xFFFFFFFF
if zip64:
    loc = tail.rfind(b"PK\x06\x07")  # zip64 EOCD locator
    if loc < 0:
        sys.exit("zip64 indicated but no locator")
    z64_eocd_offset = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
    z64 = get_range(URL, z64_eocd_offset, z64_eocd_offset + 56 - 1)
    # zip64 EOCD: sig(4) size(8) vermade(2) verneed(2) disk(4) cddisk(4)
    # entries_disk(8) total_entries(8) cd_size(8) cd_offset(8)
    total_entries = struct.unpack("<Q", z64[32:40])[0]
    cd_size = struct.unpack("<Q", z64[40:48])[0]
    cd_offset = struct.unpack("<Q", z64[48:56])[0]
    print(f"zip64 EOCD: entries={total_entries} cd_size={cd_size} cd_offset={cd_offset}")

# Download the central directory only.
print(f"fetching central directory: {cd_size/1e6:.1f} MB at offset {cd_offset}")
cd = get_range(URL, cd_offset, cd_offset + cd_size - 1)

# Parse central directory headers (sig 0x02014b50) for ALL names + sizes.
i = 0
entries = []
while i + 46 <= len(cd):
    if cd[i:i + 4] != b"PK\x01\x02":
        break
    comp_size = struct.unpack("<I", cd[i + 20:i + 24])[0]
    uncomp_size = struct.unpack("<I", cd[i + 24:i + 28])[0]
    name_len = struct.unpack("<H", cd[i + 28:i + 30])[0]
    extra_len = struct.unpack("<H", cd[i + 30:i + 32])[0]
    comment_len = struct.unpack("<H", cd[i + 32:i + 34])[0]
    name = cd[i + 46:i + 46 + name_len].decode("utf-8", "replace")
    entries.append((name, comp_size, uncomp_size))
    i += 46 + name_len + extra_len + comment_len

print(f"\nTOP-LEVEL ENTRIES ({len(entries)}):")
for name, comp, uncomp in entries:
    cs = comp / 1e6
    us = uncomp / 1e6
    note = "  <-- TARGET" if "MH_01" in name else ""
    print(f"  {name:40s} comp={cs:8.1f} MB  uncomp={us:8.1f} MB{note}")


from __future__ import annotations

from pathlib import Path

from .data import IntrinsicsEstimate


def pinhole_camera_line(
    estimate: IntrinsicsEstimate,
    *,
    camera_id: int,
    width: int,
    height: int,
) -> str:
    fx = estimate.focal if estimate.fx is None else estimate.fx
    fy = estimate.focal if estimate.fy is None else estimate.fy
    return (
        f"{int(camera_id)} PINHOLE {int(width)} {int(height)} "
        f"{fx:.17g} {fy:.17g} {estimate.cx:.17g} {estimate.cy:.17g}"
    )


def write_single_pinhole_cameras_txt(
    output_path: Path,
    estimate: IntrinsicsEstimate,
    *,
    camera_id: int = 1,
    width: int,
    height: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"{pinhole_camera_line(estimate, camera_id=camera_id, width=width, height=height)}\n"
    )


def replace_pinhole_camera_in_txt(
    input_path: Path,
    output_path: Path,
    estimate: IntrinsicsEstimate,
    *,
    camera_id: int,
    width: int | None = None,
    height: int | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    lines = input_path.read_text().splitlines()
    replaced = False
    output_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output_lines.append(line)
            continue
        parts = stripped.split()
        if int(parts[0]) != int(camera_id):
            output_lines.append(line)
            continue
        model = parts[1].upper()
        if model != "PINHOLE":
            raise ValueError(f"Only PINHOLE cameras are supported here, got {model!r}.")
        out_width = int(parts[2]) if width is None else int(width)
        out_height = int(parts[3]) if height is None else int(height)
        output_lines.append(
            pinhole_camera_line(
                estimate,
                camera_id=camera_id,
                width=out_width,
                height=out_height,
            )
        )
        replaced = True
    if not replaced:
        raise ValueError(f"Camera id {camera_id} was not found in {input_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n")

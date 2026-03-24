from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def zip_paths(output_zip: Path, paths: list[Path], base_dir: Path | None = None) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        for path in paths:
            if not path.exists():
                continue
            if path.is_dir():
                for item in path.rglob("*"):
                    if item.is_file():
                        arcname = item.relative_to(base_dir) if base_dir and item.is_relative_to(base_dir) else item.name
                        zf.write(item, arcname=str(arcname))
            else:
                arcname = path.relative_to(base_dir) if base_dir and path.is_relative_to(base_dir) else path.name
                zf.write(path, arcname=str(arcname))
    return output_zip

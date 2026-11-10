"""Build dist/regime.zip for Spark ``--py-files``, with forward-slash entry names."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "src"
    target = ROOT / "dist" / "regime.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((source / "regime").rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                archive.write(path, path.relative_to(source).as_posix())
    print(target)


if __name__ == "__main__":
    main()

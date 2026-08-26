"""Zip a run directory and, on Colab, trigger the browser download.

Usage:
    python colab/export.py runs/<run-id>

Writes runs/<run-id>.zip next to the run dir. Inside a Colab notebook runtime
the file is also pushed to the browser via google.colab.files.download; outside
Colab (local machine) the zip is created silently so you can sync it yourself.
"""

import argparse
import shutil
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"not a directory: {run_dir}")
    parent = run_dir.parent
    zip_path = parent / f"{run_dir.name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=parent,
                        base_dir=run_dir.name)
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    try:
        from google.colab import files  # only importable inside a Colab runtime

        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        files.download(str(zip_path))
        print("downloaded via Colab browser prompt")
    else:
        print("not on Colab: grab the zip from the file browser / scp it yourself")


if __name__ == "__main__":
    main()
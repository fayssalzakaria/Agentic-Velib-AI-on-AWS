"""
control.py — Marqueurs d'exécution dans le lakehouse (lake/_control/).

Sert au mode principal / secours du job quotidien : Airflow écrit le marqueur
après un job réussi ; GitHub Actions ne relance pas le job si le marqueur existe.

    python -m pipeline.control is-done daily     # code retour 0 si déjà fait aujourd'hui
    python -m pipeline.control mark-done daily
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from pyarrow import fs as pa_fs

from pipeline.settings import AWS_REGION, PARIS_TIMEZONE, is_s3, lake_root, orchestrator


def _marker_path(job: str, day: str) -> tuple[pa_fs.FileSystem, str]:
    root = f"{lake_root()}/_control/{job}"
    if is_s3(root):
        return pa_fs.S3FileSystem(region=AWS_REGION), f"{root.removeprefix('s3://')}/{day}.json"
    return pa_fs.LocalFileSystem(), os.path.abspath(f"{root}/{day}.json")


def today() -> str:
    return pd.Timestamp.now(tz=PARIS_TIMEZONE).strftime("%Y-%m-%d")


def done_by(job: str, day: str | None = None) -> str | None:
    """Orchestrateur qui a terminé le job ce jour-là, sinon None."""
    filesystem, path = _marker_path(job, day or today())
    if filesystem.get_file_info(path).type == pa_fs.FileType.NotFound:
        return None
    with filesystem.open_input_stream(path) as stream:
        return json.loads(stream.read()).get("orchestrator", "inconnu")


def mark_done(job: str, day: str | None = None) -> str:
    filesystem, path = _marker_path(job, day or today())
    parent = path.rsplit("/", 1)[0] if is_s3(lake_root()) else os.path.dirname(path)
    filesystem.create_dir(parent, recursive=True)
    payload = {"orchestrator": orchestrator(), "finished_at": datetime.now(timezone.utc).isoformat()}
    with filesystem.open_output_stream(path, compression=None) as out:
        out.write(json.dumps(payload).encode("utf-8"))
    return path


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"is-done", "mark-done"}:
        sys.exit("usage : python -m pipeline.control {is-done|mark-done} <job>")
    command, job = sys.argv[1], sys.argv[2]

    if command == "mark-done":
        print(f"Marqueur écrit : {mark_done(job)}")
        return

    by = done_by(job)
    if by:
        print(f"{job} déjà fait aujourd'hui par {by}")
        sys.exit(0)
    print(f"{job} pas encore fait aujourd'hui")
    sys.exit(1)


if __name__ == "__main__":
    main()

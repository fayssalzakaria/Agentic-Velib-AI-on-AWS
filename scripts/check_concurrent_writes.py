"""
check_concurrent_writes.py — Vérifie sur le VRAI bucket S3 que deux écrivains
simultanés (Airflow + GitHub Actions) ne perdent aucune écriture Delta.

Lance N processus qui écrivent chacun un snapshot différent, au même instant,
dans une table de test isolée (lake/_tests/...), puis vérifie :
- qu'aucune écriture annoncée comme réussie n'a disparu (perte silencieuse = grave) ;
- combien d'écritures ont été refusées avec une erreur (visible, rejouable).

    LAKE_ROOT=s3://<bucket>/lake python scripts/check_concurrent_writes.py
    LAKE_ROOT=s3://<bucket>/lake python scripts/check_concurrent_writes.py --cleanup

Identifiants : ceux du profil AWS local (droits d'écriture sur lake/*).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECORD = {
    "stationcode": "00000", "name": "STATION TEST", "capacity": 10,
    "numbikesavailable": 3, "numdocksavailable": 7, "mechanical": 2, "ebike": 1,
}


def _test_root(lake_root: str) -> str:
    return f"{lake_root.rstrip('/')}/_tests/concurrency"


def _writer(index: int, test_root: str, start_at: float, results):
    os.environ["LAKE_ROOT"] = test_root
    os.environ["PIPELINE_ORCHESTRATOR"] = f"writer-{index}"
    from pipeline.lake import merge_silver
    from pipeline.transform import to_silver

    ts = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)
    table = to_silver([RECORD], ts)
    time.sleep(max(0.0, start_at - time.time()))  # départ synchronisé
    try:
        results[index] = merge_silver(table)
    except Exception as e:  # noqa: BLE001 — on veut voir toute erreur
        results[index] = f"ERREUR : {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="supprime la table de test")
    parser.add_argument("--writers", type=int, default=2, help="écrivains simultanés (2 = Airflow + Actions)")
    parser.add_argument("--allow-local", action="store_true", help="dev : teste le script sur disque local")
    args = parser.parse_args()

    test_root = _test_root(os.environ.get("LAKE_ROOT", ""))
    on_s3 = test_root.startswith("s3://")
    if not on_s3 and not args.allow_local:
        sys.exit("LAKE_ROOT doit pointer vers S3 (s3://<bucket>/lake).")

    from pyarrow import fs as pa_fs
    from pipeline.settings import AWS_REGION

    if on_s3:
        s3, test_path = pa_fs.S3FileSystem(region=AWS_REGION), test_root.removeprefix("s3://")
    else:
        s3, test_path = pa_fs.LocalFileSystem(), os.path.abspath(test_root)

    if args.cleanup:
        s3.delete_dir(test_path)
        print(f"Supprimé : {test_path}")
        return

    # Table de test vide au départ, créée avant la course
    try:
        s3.delete_dir(test_path)
    except (FileNotFoundError, OSError):
        pass
    os.environ["PIPELINE_ORCHESTRATOR"] = "setup"
    os.environ["LAKE_ROOT"] = test_root
    from pipeline.lake import merge_silver
    from pipeline.transform import to_silver
    merge_silver(to_silver([RECORD], datetime(1999, 12, 31, 23, tzinfo=timezone.utc)))

    with mp.Manager() as manager:
        results = manager.dict()
        start_at = time.time() + 5
        procs = [mp.Process(target=_writer, args=(i, test_root, start_at, results)) for i in range(args.writers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        results = dict(results)

    from deltalake import DeltaTable
    from pipeline.settings import delta_storage_options
    dt = DeltaTable(f"{test_root}/silver/station_status", storage_options=delta_storage_options())
    present = set(dt.to_pyarrow_table(columns=["ingested_by"]).column("ingested_by").to_pylist())

    succeeded = [i for i, r in results.items() if isinstance(r, int)]
    rejected = {i: r for i, r in results.items() if not isinstance(r, int)}
    silently_lost = [i for i in succeeded if f"writer-{i}" not in present]

    print(f"Versions Delta : {dt.version()}")
    print(f"Écritures réussies : {len(succeeded)} / {args.writers}")
    for i, error in rejected.items():
        print(f"Écrivain {i} refusé avec une erreur (rejouable) : {error.splitlines()[0]}")
    if silently_lost:
        sys.exit(f"ÉCHEC : écritures annoncées réussies mais absentes : {silently_lost}")
    if rejected:
        sys.exit("ATTENTION : aucune perte silencieuse, mais des écritures ont épuisé leurs tentatives.")
    print("OK : toutes les écritures concurrentes sont présentes, aucune perte.")


if __name__ == "__main__":
    main()

# utils/case_logger.py
import os
import json
import datetime
import threading
from typing import Any, Dict, Iterable, Optional


def _to_list(x):
    """Convert torch.Tensor / np.ndarray to Python lists (else passthrough)."""
    if x is None:
        return None
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        if isinstance(x, np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    return x


def _to_float(x):
    """Best-effort float conversion (None on failure)."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


class CaseLogger:
    """
    Append per-test results as JSONL (one JSON object per line).

    Usage:
        logger = CaseLogger(".../cases.jsonl")              # overwrite by default
        logger.log({"case_id": 7, "regret": 0.00123})       # dict style
        logger.log(case_id=8, regret=0.00456, algo="pno")   # kwargs style
        logger.close()

        # or as a context manager:
        with CaseLogger(".../cases.jsonl", mode="w") as cl:
            cl.log(case_id=0, regret=0.0)
    """

    def __init__(self, out_jsonl_path: str, mode: str = "w"):
        """
        Args:
            out_jsonl_path: destination .jsonl file
            mode: "w" to overwrite each run (default), or "a" to append
        """
        self.out = out_jsonl_path
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        # line-buffered text file; flush after each write
        self._fh = open(self.out, mode, buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, record: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Log a single record. Accepts either a dict or keyword args (or both).
        When both are given, kwargs override dict keys.
        """
        if record is None:
            record = {}
        elif not isinstance(record, dict):
            raise TypeError("CaseLogger.log expects a dict (or use keyword args).")

        if kwargs:
            record.update(kwargs)

        # Normalize common numeric fields if present
        for k in ("regret", "rel_regret", "mse", "mae"):
            if k in record:
                record[k] = _to_float(record[k])

        # Normalize possible array-like fields if present
        for k in ("pred", "real", "alloc", "oracle_alloc"):
            if k in record:
                record[k] = _to_list(record[k])

        # Optional standard fields
        if "algo" in record and record["algo"] is not None:
            record["algo"] = str(record["algo"])

        if "time" in record and record["time"] is not None:
            # leave as provided (string or number); caller decides
            pass

        # Stamp the log time if not provided
        record.setdefault("logged_at", datetime.datetime.utcnow().isoformat() + "Z")

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def log_many(self, records: Iterable[Dict[str, Any]]) -> None:
        """Log multiple records."""
        for r in records:
            self.log(r)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
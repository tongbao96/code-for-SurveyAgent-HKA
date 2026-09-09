from __future__ import annotations

import re
from datetime import datetime
from threading import Lock
from typing import Callable, Optional


ProgressCallback = Optional[Callable[[str], None]]
_BAR_PATTERN = re.compile(r"^\[progress\] (.+?) :: (\d+)/(\d+)$")
_CONSOLE_LOCK = Lock()
_ACTIVE_BAR_WIDTH = 0


def report(callback: ProgressCallback, message: str) -> None:
    if callback:
        callback(message)


def report_bar(
    callback: ProgressCallback, label: str, done: int, total: int
) -> None:
    if callback and total > 0:
        callback(f"[progress] {label} :: {min(max(done, 0), total)}/{total}")


def console_progress(message: str) -> None:
    global _ACTIVE_BAR_WIDTH
    timestamp = datetime.now().strftime("%H:%M:%S")
    progress = _BAR_PATTERN.match(message)
    with _CONSOLE_LOCK:
        if progress:
            label, done_text, total_text = progress.groups()
            done, total = int(done_text), max(1, int(total_text))
            width = 28
            filled = min(width, int(width * done / total))
            text = (
                f"[{timestamp}] {label} "
                f"[{'#' * filled}{'-' * (width - filled)}] "
                f"{done}/{total} ({100 * done // total:3d}%)"
            )
            print(
                "\r" + text.ljust(_ACTIVE_BAR_WIDTH),
                end="\n" if done >= total else "\r",
                flush=True,
            )
            _ACTIVE_BAR_WIDTH = 0 if done >= total else len(text)
            return

        if _ACTIVE_BAR_WIDTH:
            print("\r" + " " * _ACTIVE_BAR_WIDTH + "\r", end="", flush=True)
            _ACTIVE_BAR_WIDTH = 0
        stage = re.match(r"^\[(\d+)/(\d+)\]", message)
        starts_stage = stage and (
            "started" in message
            or "loaded checkpoint" in message
            or "skipped" in message
        )
        if starts_stage:
            print("\n" + "=" * 88, flush=True)
        print(f"[{timestamp}] {message}", flush=True)
        if starts_stage:
            print("=" * 88, flush=True)


def at_milestone(done: int, total: int, updates: int = 10) -> bool:
    if total <= 0:
        return False
    interval = max(1, total // updates)
    return done == 1 or done == total or done % interval == 0

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class RunStore:
    """Files written by one pipeline run."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"

    def save(self, name: str, value: Any) -> Path:
        path = self.artifacts / f"{name}.json"
        temp = path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temp, path)
        return path

    def load(self, name: str, default: Any = None) -> Any:
        path = self.artifacts / f"{name}.json"
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_text(self, name: str, text: str, extension: str = ".md") -> Path:
        path = self.artifacts / f"{name}{extension}"
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
        return path

    def state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"completed": [], "current": None}
        with self.state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def reset_state(self) -> None:
        """Forget checkpoint status while leaving previous artifacts on disk."""
        self._write_state({"completed": [], "current": None, "artifacts": {}})
        self.event("pipeline", "checkpoint state reset")

    def mark_started(self, stage: str) -> None:
        state = self.state()
        # A failed --no-resume run must not leave an older artifact marked as a
        # valid checkpoint for the stage that is currently being recomputed.
        state["completed"] = [item for item in state.get("completed", []) if item != stage]
        state.setdefault("artifacts", {}).pop(stage, None)
        state["current"] = stage
        self._write_state(state)
        self.event("pipeline", "started", {"stage": stage})

    def mark_completed(self, stage: str, artifact: Optional[str] = None) -> None:
        state = self.state()
        if stage not in state["completed"]:
            state["completed"].append(stage)
        state["current"] = None
        if artifact:
            state.setdefault("artifacts", {})[stage] = artifact
        self._write_state(state)
        self.event("pipeline", "completed", {"stage": stage, "artifact": artifact})

    def is_completed(self, stage: str) -> bool:
        return stage in self.state().get("completed", [])

    def event(
        self, component: str, action: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "action": action,
            "payload": payload or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_state(self, state: Dict[str, Any]) -> None:
        temp = self.state_path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(temp, self.state_path)

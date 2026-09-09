from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

from .progress import ProgressCallback, report


AgentTask = Dict[str, Any]
AgentEvent = Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]]


def run_agent_tasks(
    agent: str,
    tasks: Sequence[AgentTask],
    worker: Callable[[Any], Any],
    max_workers: int,
    progress: ProgressCallback = None,
    event: AgentEvent = None,
    continue_on_error: bool = False,
) -> List[Any]:
    """Run independent agent tasks with bounded concurrency and ordered output."""
    if not tasks:
        return []

    workers = max(1, min(int(max_workers), len(tasks)))
    report(
        progress,
        f"[ROUTER] dispatching {len(tasks)} task(s) to [{agent}] "
        f"with {workers} worker(s)",
    )
    _event(event, "router", "dispatch", {
        "agent": agent,
        "sender": "ROUTER",
        "receiver": agent,
        "tasks": len(tasks),
        "workers": workers,
    })

    results: List[Any] = [None] * len(tasks)
    if workers == 1:
        for index, task in enumerate(tasks):
            try:
                results[index] = _run_one(
                    agent, task, worker, index, len(tasks), progress, event
                )
            except RuntimeError:
                if not continue_on_error:
                    raise
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, task in enumerate(tasks):
            label = _label(task, index)
            report(progress, f"[{agent}] queued {index + 1}/{len(tasks)}: {label}")
            _event(event, agent.lower(), "queued", {
                **_task_payload(task, index),
                "sender": "ROUTER",
                "receiver": agent,
            })
            futures[executor.submit(worker, task.get("payload"))] = (index, task)

        completed = 0
        try:
            for future in as_completed(futures):
                index, task = futures[future]
                label = _label(task, index)
                try:
                    results[index] = future.result()
                except Exception as exc:
                    _event(event, agent.lower(), "failed", {
                        **_task_payload(task, index),
                        "sender": agent,
                        "receiver": "ROUTER",
                        "error_type": type(exc).__name__,
                    })
                    report(progress, f"[{agent}] failed: {label} ({type(exc).__name__})")
                    if continue_on_error:
                        completed += 1
                        continue
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"[{agent}] task failed: {label}") from exc
                completed += 1
                _event(event, agent.lower(), "completed", {
                    **_task_payload(task, index),
                    "sender": agent,
                    "receiver": "ROUTER",
                })
                report(
                    progress,
                    f"[{agent}] completed {completed}/{len(tasks)}: {label}",
                )
        finally:
            for pending in futures:
                pending.cancel()
    return results


def configured_workers(enabled: bool, requested: int) -> int:
    return max(1, int(requested)) if enabled else 1


def _run_one(
    agent: str,
    task: AgentTask,
    worker: Callable[[Any], Any],
    index: int,
    total: int,
    progress: ProgressCallback,
    event: AgentEvent,
) -> Any:
    label = _label(task, index)
    report(progress, f"[{agent}] started {index + 1}/{total}: {label}")
    _event(event, agent.lower(), "started", {
        **_task_payload(task, index),
        "sender": "ROUTER",
        "receiver": agent,
    })
    try:
        result = worker(task.get("payload"))
    except Exception as exc:
        _event(event, agent.lower(), "failed", {
            **_task_payload(task, index),
            "sender": agent,
            "receiver": "ROUTER",
            "error_type": type(exc).__name__,
        })
        report(progress, f"[{agent}] failed: {label} ({type(exc).__name__})")
        raise RuntimeError(f"[{agent}] task failed: {label}") from exc
    _event(event, agent.lower(), "completed", {
        **_task_payload(task, index),
        "sender": agent,
        "receiver": "ROUTER",
    })
    report(progress, f"[{agent}] completed {index + 1}/{total}: {label}")
    return result


def _label(task: AgentTask, index: int) -> str:
    return str(task.get("label") or task.get("id") or f"task-{index + 1}")


def _task_payload(task: AgentTask, index: int) -> Dict[str, Any]:
    return {
        "task_id": str(task.get("id") or f"task-{index + 1}"),
        "label": _label(task, index),
        "position": index + 1,
    }


def _event(
    callback: AgentEvent,
    component: str,
    action: str,
    payload: Dict[str, Any],
) -> None:
    if callback:
        callback(component, action, payload)

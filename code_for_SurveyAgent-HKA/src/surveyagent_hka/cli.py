from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import PipelineConfig
from .llm import OpenAIClient
from .pipeline import run_pipeline
from .progress import console_progress
from .retrieval import build_sources


DEFAULT_CONFIG = "config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surveyagent-hka", description="SurveyAgent-HKA scientific survey generation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the end-to-end pipeline")
    run.add_argument("--config", default=DEFAULT_CONFIG, help="Configuration file (default: config.json)")
    run.add_argument("--no-resume", action="store_true", help="Recompute stages even if checkpoints exist")
    status = commands.add_parser("status", help="Show checkpoint state")
    status.add_argument("--config", default=DEFAULT_CONFIG)
    check = commands.add_parser("validate-config", help="Load and validate configuration without running agents")
    check.add_argument("--config", default=DEFAULT_CONFIG)
    probe = commands.add_parser("test-llm", help="Send one minimal JSON request to the configured LLM")
    probe.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig.load(args.config)
    if args.command == "validate-config":
        build_sources(config.retrieval, console_progress)
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "test-llm":
        print("[1/4] Configuration loaded")
        api_key = config.llm_api_key()
        print("[2/4] API key found in config.json (value hidden)")
        model = config.model.model
        base_url = config.model.base_url
        if not base_url:
            raise SystemExit("No model.base_url configured in config.json")
        print(f"[3/4] Calling {base_url} with model {model}")
        llm = OpenAIClient(config.model, api_key, base_url)
        result = llm.json(
            'Reply with one JSON object only: {"status":"ok"}. '
            'Do not add Markdown or explanation.'
        )
        if str(result.get("status", "")).lower() != "ok":
            raise RuntimeError(f"Unexpected LLM response: {result}")
        print("[4/4] PASS - KKSJ endpoint, key, model, and JSON output are working")
        return 0
    if args.command == "status":
        state_path = Path(config.workspace) / "state.json"
        print(state_path.read_text(encoding="utf-8") if state_path.exists() else '{"completed": [], "current": null}')
        return 0
    if args.command == "run":
        final = run_pipeline(
            config, resume=not args.no_resume, progress=console_progress
        )
        print(f"Survey complete: {Path(config.workspace) / 'artifacts' / 'final_survey.md'}")
        print(f"Sections: {len(final.sections)} | References: {len(final.bibliography)} | Revision round: {final.revision_round}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

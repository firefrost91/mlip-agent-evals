"""Helpers for pulling traces out of Phoenix into pandas. You should not need to edit this file."""

from __future__ import annotations

import os
import re

import pandas as pd
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery

PHOENIX = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
SWEBENCH_PROJECT = "lab11-swebench"


def client() -> Client:
    return Client(base_url=PHOENIX)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Phoenix returns custom attributes as nested dicts (attributes.agent = {...}); expand them into columns."""
    df = df.copy()
    for col in [c for c in df.columns if c.startswith("attributes.")]:
        prefix = col[len("attributes."):]
        if df[col].map(lambda v: isinstance(v, dict)).any():
            expanded = pd.json_normalize(df[col].map(lambda v: v if isinstance(v, dict) else {}).tolist())
            expanded.index = df.index
            expanded.columns = [f"{prefix}.{c}" for c in expanded.columns]
            df = df.drop(columns=[col]).join(expanded)
        else:
            df = df.rename(columns={col: prefix})
    df = df.drop(columns=["context.span_id"], errors="ignore")
    df.index.name = "span_id"
    return df.rename(columns={"input.value": "input", "output.value": "output", "context.trace_id": "otel_trace_id"})


def spans(project: str = SWEBENCH_PROJECT, where: str | None = None, limit: int = 100_000) -> pd.DataFrame:
    """All spans in a project (optionally filtered with the same query language as Phoenix's filter bar)."""
    query = SpanQuery().where(where) if where else None
    df = client().spans.get_spans_dataframe(project_identifier=project, query=query, limit=limit)
    return _flatten(df).sort_values("start_time")


def runs(project: str = SWEBENCH_PROJECT) -> pd.DataFrame:
    """One row per agent run (the root agent.run span), indexed by span id."""
    return spans(project, where="parent_id is None")


def tool_calls(project: str = SWEBENCH_PROJECT) -> pd.DataFrame:
    """Every bash command, in order, with the otel_trace_id of the run it belongs to."""
    return spans(project, where="span_kind == 'TOOL'")


def annotations(name: str, project: str = SWEBENCH_PROJECT) -> pd.DataFrame:
    """Annotations called `name` on run root spans, indexed by span id: label, score, explanation."""
    r = client().spans.get_spans_dataframe(project_identifier=project, query=SpanQuery().where("parent_id is None"),
                                           limit=100_000)
    ann = client().spans.get_span_annotations_dataframe(spans_dataframe=r, project_identifier=project)
    ann = ann[ann.annotation_name == name]
    return ann.rename(columns={"result.label": "label", "result.score": "score",
                               "result.explanation": "explanation"})[["label", "score", "explanation"]]


def annotate(labels: pd.Series, name: str, explanations: pd.Series | None = None, kind: str = "CODE",
             project: str = SWEBENCH_PROJECT):
    """Attach a label to each span (labels is indexed by span id). It then shows up in the Phoenix UI and can be
    used in filters, e.g.  annotations['failure_mode'].label == 'wrong fix, self-check passed'"""
    df = pd.DataFrame({"label": labels.astype(str)})
    if explanations is not None:
        df["explanation"] = explanations.astype(str)
    df.index.name = "span_id"
    client().spans.log_span_annotations_dataframe(dataframe=df, annotation_name=name, annotator_kind=kind, sync=True)
    print(f"attached '{name}' to {len(df)} spans")


# ---------------------------------------------------------------- shell-command heuristics

_EDIT = [r"\bsed\s+-i", r"\bperl\s+-pi", r"cat\s*>\s*\S+\.py", r"cat\s+<<.*>\s*\S+\.py", r">\s*\S+\.py\b",
         r"\.write_text\(", r"open\([^)]*['\"]w['\"]", r"\bpatch\s+-p", r"\bgit\s+apply\b"]
_TEST = [r"\bpytest\b", r"-m\s+pytest", r"\btox\b", r"runtests\.py", r"-m\s+unittest",
         r"\bpython3?\s+\S*(reproduce|repro|test)\S*\.py", r"\bpython3?\s+-c\s"]


def is_edit_command(cmd: str) -> bool:
    """Does this shell command (probably) modify a source file?"""
    if "git diff" in cmd and "patch.txt" in cmd:
        return False
    return any(re.search(p, cmd) for p in _EDIT)


def is_test_command(cmd: str) -> bool:
    """Does this shell command (probably) run tests or a reproduction script?"""
    return any(re.search(p, cmd) for p in _TEST)


def patch_touches_tests(patch: str) -> bool:
    files = re.findall(r"^\+\+\+ b/(\S+)", patch or "", flags=re.M)
    return any(re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_tests?\.py$|conftest\.py$", f) for f in files)


def part1_run() -> str:
    """The run used in Part 1 (configured in data/lab_config.json)."""
    return lab_config()["part1_run"]


def lab_config() -> dict:
    import json
    from pathlib import Path

    return json.loads((Path(__file__).parent / "data" / "lab_config.json").read_text())


def model_results() -> pd.DataFrame:
    """Public SWE-bench Verified results for several models (same agent, all 500 tasks): resolved and $ per task."""
    from pathlib import Path

    return pd.read_csv(Path(__file__).parent / "data" / "model_results.csv")


def all_runs() -> pd.DataFrame:
    """Every run of the agent model on SWE-bench Verified (not just the ones loaded into Phoenix), with
    per-step token counts: one row per run."""
    from pathlib import Path

    return pd.read_json(Path(__file__).parent / "data" / "all_runs.jsonl", lines=True)


# ---------------------------------------------------------------- Part 5: the judge's Phoenix experiments

JUDGE_DATASET = "lab11-judge-set"


def judge_experiments() -> pd.DataFrame:
    """Every experiment you've run on the judge dataset (each Playground run over the dataset is one)."""
    c = client()
    ds = c.datasets.get_dataset(dataset=JUDGE_DATASET)
    r = c._client.get(f"v1/datasets/{ds.id}/experiments")
    r.raise_for_status()
    cols = ["name", "id", "created_at", "successful_run_count", "failed_run_count"]
    return pd.DataFrame(r.json()["data"], columns=cols).sort_values("created_at")


def judge_results(experiment: str) -> pd.DataFrame:
    """One row per run for a judge experiment (by name; the latest one if several share a name):
    what the judge said, what the hidden tests say, and the judge's full answer.

    experiment="precomputed" loads data/judge_outputs.jsonl instead (offline fallback, no API key needed)."""
    import json
    from pathlib import Path

    from judge import parse_verdict

    if experiment == "precomputed":
        path = Path(__file__).parent / "data" / "judge_outputs.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} isn't there: precomputed verdicts are only available if your TA "
                                    "provides that file. Ask them, or set up a Gemini key for Part 5.")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        truth = {t["trace_id"]: t["resolved"] for t in map(json.loads, (Path(__file__).parent / "data" /
                                                                       "traces.jsonl").read_text().splitlines())}
        return pd.DataFrame({"agent.trace_id": [r["trace_id"] for r in rows],
                             "judge_says": [("resolved" if r["judge_says_resolved"] else "not_resolved") for r in rows],
                             "hidden_tests": [("resolved" if truth[r["trace_id"]] else "not_resolved") for r in rows],
                             "judge_output": [r["judge_reason"] for r in rows]}).set_index("agent.trace_id")

    exps = judge_experiments()
    match = exps[exps["name"] == experiment]
    if match.empty:
        raise ValueError(f"no experiment named {experiment!r}; you have: {list(exps['name'])}")
    c = client()
    r = c._client.get(f"v1/experiments/{match.iloc[-1]['id']}/json")
    r.raise_for_status()
    examples = {e["id"]: e["metadata"]["agent.trace_id"]
                for e in c.datasets.get_dataset(dataset=JUDGE_DATASET).examples}
    rows = []
    for run in r.json():
        msgs = (run.get("output") or {}).get("messages") or [{}]
        text = msgs[-1].get("content") or ""
        rows.append({"agent.trace_id": examples.get(run["example_id"], run["example_id"]),
                     "judge_says": parse_verdict(text) or "no verdict",
                     "hidden_tests": run["reference_output"]["verdict"],
                     "judge_output": text, "error": run.get("error"),
                     "prompt_tokens": run.get("prompt_token_count")})
    return pd.DataFrame(rows).set_index("agent.trace_id")

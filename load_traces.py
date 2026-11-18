"""Load recorded agent runs (data/traces.jsonl) into Phoenix as OpenTelemetry traces.

Each run becomes one trace with the same shape your own instrumentation produces in Part 1:

    agent.run (AGENT)                      one SWE-bench task, one attempt
    └─ agent.step N (CHAIN)
       ├─ llm (LLM)                        the model's reasoning for that step
       └─ tool.bash (TOOL)                 the command it ran, its output and exit code

Each run's session id is its SWE-bench instance id (this data has one attempt per task). Successful commands get
status OK and failing ones (non-zero exit code) ERROR. The script also sets up Part 5 in Phoenix: the judge
dataset, prompt and evaluator.

    python load_traces.py                  # into project "lab11-swebench" on http://localhost:6006
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

DATA = Path(__file__).parent / "data"
SEC = 1_000_000_000


def _span(tracer, name, parent_ctx, start, end, kind, attrs, error=None, ok=False):
    span = tracer.start_span(name, context=parent_ctx, start_time=start)
    span.set_attribute("openinference.span.kind", kind)
    for k, v in attrs.items():
        if v is not None:
            span.set_attribute(k, v)
    if error:
        span.set_status(Status(StatusCode.ERROR, error))
    elif ok:
        span.set_status(Status(StatusCode.OK))
    span.end(end_time=end)
    return span


def _llm_attrs(t: dict, s: dict) -> dict:
    """OpenInference attributes for one model call: model, its own token counts and cost."""
    tin, tout = s.get("input_tokens"), s.get("output_tokens")
    return {
        "llm.model_name": t["model"],
        "output.value": s.get("thought", ""),
        "llm.token_count.prompt": tin,
        "llm.token_count.completion": tout,
        "llm.token_count.total": (tin or 0) + (tout or 0) if tin is not None else None,
        "llm.cost.total": s.get("cost"),
        "agent.step_index": s["i"],
    }


def load_autotrace_only(t: dict, tracer):
    """The same run as off-the-shelf LiteLLM auto-instrumentation records it: one trace per model call, with the
    call's messages, but no agent loop, no step structure, and no spans for the commands themselves.

    Real auto-instrumentation records the model's whole message history on every call. To keep the data small we
    keep only the newest input message: the task on the first call, then the previous command's output."""
    clock = time.time_ns() - 2 * 24 * 3600 * SEC
    newest = t["problem_statement"]
    for s in t["steps"]:
        span = tracer.start_span("completion", start_time=clock)
        span.set_attribute("openinference.span.kind", "LLM")
        for k, v in _llm_attrs(t, s).items():
            if v is not None and k not in ("agent.step_index", "output.value"):
                span.set_attribute(k, v)
        span.set_attribute("input.value", newest[:4000])
        span.set_attribute("llm.input_messages.0.message.role", "user")
        span.set_attribute("llm.input_messages.0.message.content", newest[:4000])
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute("llm.output_messages.0.message.content", s.get("thought") or "")
        for j, a in enumerate(s.get("actions") or []):
            prefix = f"llm.output_messages.0.message.tool_calls.{j}.tool_call.function"
            span.set_attribute(f"{prefix}.name", "bash")
            span.set_attribute(f"{prefix}.arguments", json.dumps({"command": a["command"]}))
        span.set_attribute("output.value", (s.get("thought") or "") + "".join(
            f"\n[tool call] bash: {a['command']}" for a in s.get("actions") or []))
        if s.get("actions"):
            newest = "\n".join(f"<returncode>{a['returncode']}</returncode>\n<output>\n{(a['output'] or '')[:3000]}"
                                f"\n</output>" for a in s["actions"])
        clock += 4 * SEC
        span.end(end_time=clock)
        clock += 2 * SEC


def load_runs(traces: list[dict], tracer) -> dict[str, str]:
    """Emit one OTel trace per run. Returns {lab trace_id: root span id (hex)}."""
    roots = {}
    clock = time.time_ns() - 3 * 24 * 3600 * SEC  # runs "happened" a few days ago, spaced out
    for t in traces:
        n = max(len(t["steps"]), 1)
        per_step = int((t.get("wall_time_s") or 5 * n) / n * SEC)
        start = clock
        clock += per_step * n + 60 * SEC

        root = tracer.start_span("agent.run", start_time=start)
        root_ctx = trace.set_span_in_context(root)
        for k, v in {
            "openinference.span.kind": "AGENT",
            "session.id": t["instance_id"],
            "input.value": t["problem_statement"],
            "output.value": t["patch"] or "(empty patch)",
            "agent.trace_id": t["trace_id"],
            "agent.instance_id": t["instance_id"],
            "agent.repo": t["repo"],
            "agent.attempt": t["run"],
            "agent.model": t["model"],
            "agent.exit_status": t["exit_status"],
            "agent.n_steps": t["n_steps"],
            "agent.cost_usd": t["cost_usd"],
            "llm.token_count.prompt": t["input_tokens"],
            "llm.token_count.completion": t["output_tokens"],
            "eval.resolved": bool(t["resolved"]),
            "eval.fail_to_pass_failed": t.get("f2p_fail"),
            "eval.pass_to_pass_failed": t.get("p2p_fail"),
        }.items():
            if v is not None:
                root.set_attribute(k, v)

        cursor = start
        for s in t["steps"]:
            s_start, s_end = cursor, cursor + per_step
            cursor = s_end
            step = tracer.start_span(f"agent.step {s['i']}", context=root_ctx, start_time=s_start)
            step.set_attribute("openinference.span.kind", "CHAIN")
            step.set_attribute("agent.step_index", s["i"])
            step_ctx = trace.set_span_in_context(step)
            llm_end = s_start + per_step // 2
            llm_attrs = _llm_attrs(t, s)
            if s.get("format_error"):
                llm_attrs["output.value"] = s.get("thought") or "(no valid command)"
                _span(tracer, "llm", step_ctx, s_start, llm_end, "LLM", llm_attrs,
                      error="format error: no valid command")
            else:
                _span(tracer, "llm", step_ctx, s_start, llm_end, "LLM", llm_attrs)
                for a in s["actions"]:
                    rc = a["returncode"]
                    _span(tracer, "tool.bash", step_ctx, llm_end, s_end, "TOOL",
                          {"tool.name": "bash", "input.value": a["command"], "output.value": a["output"],
                           "tool.returncode": rc, "agent.step_index": s["i"], "agent.trace_id": t["trace_id"]},
                          error=f"exit code {rc}" if rc not in (0, None) else None, ok=rc == 0)
            step.end(end_time=s_end)
        root.end(end_time=cursor)
        roots[t["trace_id"]] = format(root.get_span_context().span_id, "016x")
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=DATA / "traces.jsonl")
    ap.add_argument("--judge", default=DATA / "judge_outputs.jsonl")
    ap.add_argument("--project", default="lab11-swebench")
    ap.add_argument("--endpoint", default=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006"))
    ap.add_argument("--force", action="store_true", help="load even if the project already has runs")
    args = ap.parse_args()

    traces = [json.loads(l) for l in Path(args.traces).read_text().splitlines() if l.strip()]
    print(f"loading {len(traces)} runs into Phoenix at {args.endpoint} (about a minute)...", flush=True)

    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

    client = Client(base_url=args.endpoint)
    try:
        existing = client.spans.get_spans_dataframe(project_identifier=args.project, limit=10_000,
                                                     query=SpanQuery().where("parent_id is None"))
    except Exception:  # project doesn't exist yet
        existing = []
    if len(existing) and not args.force:
        print(f"project '{args.project}' already has {len(existing)} runs; not loading them again (--force to reload)")
        roots = {a["trace_id"]: sid for sid, a in existing["attributes.agent"].items()}
        setup_judge(client, traces, roots, existing)
        attach_precomputed_judge(client, args.judge, roots, existing)
        return

    provider = TracerProvider(resource=Resource.create({"openinference.project.name": args.project}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{args.endpoint}/v1/traces"),
                                                max_queue_size=200_000, max_export_batch_size=512))
    roots = load_runs(traces, provider.get_tracer("lab11.loader"))
    provider.force_flush()
    print(f"sent {len(traces)} runs to project '{args.project}'")

    # Part 1: one run as auto-instrumentation alone would have recorded it, in its own project
    part1 = json.loads((DATA / "lab_config.json").read_text())["part1_run"]
    p1 = TracerProvider(resource=Resource.create({"openinference.project.name": "lab11-part1-autotrace"}))
    p1.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{args.endpoint}/v1/traces")))
    load_autotrace_only(next(t for t in traces if t["trace_id"] == part1), p1.get_tracer("lab11.loader"))
    p1.force_flush()
    print(f"sent run {part1} (auto-instrumentation only) to project 'lab11-part1-autotrace'")

    # wait until Phoenix has ingested every span (it inserts them asynchronously)
    wanted = set(roots.values())
    expected = sum(1 + sum(2 + len(s["actions"]) for s in t["steps"]) for t in traces)
    for _ in range(180):
        n = len(client.spans.get_spans_dataframe(project_identifier=args.project, limit=1_000_000))
        if n >= expected:
            break
        time.sleep(2)
    else:
        print(f"WARNING: only {n}/{expected} spans visible so far; wait a minute before starting Part 2")
    roots_df = client.spans.get_spans_dataframe(project_identifier=args.project, limit=10_000,
                                                 query=SpanQuery().where("parent_id is None"))
    print(f"{len(wanted & set(roots_df.index))} runs ({n} spans) are in Phoenix")

    setup_judge(client, traces, roots, roots_df)
    attach_precomputed_judge(client, args.judge, roots, roots_df)
    print(f"Done. Open {args.endpoint} and pick '{args.project}'.")


def attach_precomputed_judge(client, path, roots: dict[str, str], roots_df) -> None:
    """Offline fallback for Part 5: precomputed verdicts (from judge.py) as `llm_judge` annotations, if a TA
    has shipped data/judge_outputs.jsonl."""
    if not Path(path).exists():
        return
    judge = pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])
    judge["span_id"] = judge.trace_id.map(roots)
    judge = judge[judge.span_id.isin(roots_df.index)]
    ann = pd.DataFrame({
        "span_id": judge.span_id,
        "label": judge.judge_says_resolved.map({True: "resolved", False: "not_resolved"}),
        "explanation": judge.judge_reason,
    }).set_index("span_id")
    client.spans.log_span_annotations_dataframe(dataframe=ann, annotation_name="llm_judge",
                                                annotator_kind="LLM", sync=True)
    print(f"attached {len(ann)} precomputed llm_judge annotations (offline fallback for Part 5)")


def setup_judge(client, traces: list[dict], roots: dict[str, str], roots_df) -> None:
    """Part 5: the judge dataset, the starter prompt, and the evaluator that grades the judge."""
    from phoenix.client.types.prompts import PromptVersion

    from judge import DATASET_NAME, PROMPT_NAME, SYSTEM_V1, TRUE_PATTERN, USER_V1, judge_inputs

    try:
        dataset = client.datasets.get_dataset(dataset=DATASET_NAME)
        print(f"dataset '{DATASET_NAME}' already exists")
    except Exception:
        rows = []
        for t in traces:
            if roots.get(t["trace_id"]) not in roots_df.index:
                continue
            rows.append({**judge_inputs(t),
                         "verdict": "resolved" if t["resolved"] else "not_resolved",
                         "verdict_pattern": TRUE_PATTERN[bool(t["resolved"])],
                         "agent.trace_id": t["trace_id"], "model": t["model"],
                         "span_id": roots[t["trace_id"]]})
        dataset = client.datasets.create_dataset(
            name=DATASET_NAME, dataframe=pd.DataFrame(rows), input_keys=["issue", "transcript", "patch"],
            output_keys=["verdict", "verdict_pattern"], metadata_keys=["agent.trace_id", "model"],
            span_id_key="span_id",  # each example links back to the run it came from
            dataset_description="Part 5: one example per run. Input = what the judge sees; "
                                "output = the hidden tests' verdict.")
        print(f"created dataset '{DATASET_NAME}' with {len(rows)} examples")

    try:
        client.prompts.get(prompt_identifier=PROMPT_NAME)
        print(f"prompt '{PROMPT_NAME}' already exists")
    except Exception:
        model = json.loads((DATA / "lab_config.json").read_text()).get("judge_model", "gemini-3.5-flash-lite")
        client.prompts.create(name=PROMPT_NAME, prompt_description="Part 5 starter judge", version=PromptVersion(
            [{"role": "system", "content": SYSTEM_V1}, {"role": "user", "content": USER_V1}],
            model_name=model, model_provider="GOOGLE", template_format="MUSTACHE",
            description="v1: starter judge"))
        print(f"created prompt '{PROMPT_NAME}' (v1, {model})")

    _add_verdict_evaluator(client, dataset.id)


def _gql(client, query: str, variables: dict | None = None) -> dict:
    r = client._client.post("graphql", json={"query": query, "variables": variables or {}})
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(body["errors"])
    return body["data"]


def _add_verdict_evaluator(client, dataset_id: str, name: str = "verdict_correct") -> None:
    """A built-in regex evaluator: does the judge's output contain the TRUE verdict line?"""
    existing = _gql(client, """query($id: ID!) { node(id: $id) { ... on Dataset {
        datasetEvaluators(first: 50) { edges { node { name } } } } } }""", {"id": dataset_id})
    if any(e["node"]["name"] == name for e in existing["node"]["datasetEvaluators"]["edges"]):
        print(f"evaluator '{name}' already attached")
        return
    builtins = _gql(client, "{ builtInEvaluators { id name } }")["builtInEvaluators"]
    regex_id = next(e["id"] for e in builtins if e["name"] == "regex")
    _gql(client, """mutation($input: CreateDatasetBuiltinEvaluatorInput!) {
        createDatasetBuiltinEvaluator(input: $input) { evaluator { id } } }""", {"input": {
        "datasetId": dataset_id, "evaluatorId": regex_id, "name": name,
        "description": "1 if the judge's VERDICT line matches the hidden tests, else 0",
        "outputConfigs": [{"categorical": {"name": name, "optimizationDirection": "MAXIMIZE", "values": [
            {"label": "true", "score": 1.0}, {"label": "false", "score": 0.0}]}}],
        "inputMapping": {"pathMapping": {"pattern": "$.reference.verdict_pattern",
                                         "text": "$.output.messages[0].content"},
                         "literalMapping": {"full_match": False}}}})
    print(f"attached evaluator '{name}' to the dataset")

if __name__ == "__main__":
    main()

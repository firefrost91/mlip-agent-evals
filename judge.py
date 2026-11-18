"""LLM-as-judge for agent traces.

The judge reads the issue, a condensed version of the agent's trajectory, and the final patch, and predicts
whether the patch fixes the issue. It never sees the hidden SWE-bench tests, which is exactly the situation
in production: users file issues, the agent opens PRs, and nobody hands you a ground-truth test suite.

In Part 5 you run and improve this judge inside Phoenix (Prompt Playground + Experiments). `load_traces.py`
puts everything Phoenix needs there for you:
  - the dataset `lab11-judge-set`: one example per run (issue, transcript, patch -> true verdict),
  - the prompt `lab11-judge` (version 1 = the starter prompt below),
  - the dataset evaluator `verdict_correct`, which checks the judge's VERDICT line against the hidden tests.

This file is the same judge as plain Python. TAs use it to precompute verdicts as an offline fallback:
    python judge.py --model gemini/gemini-3.5-flash-lite   # needs GEMINI_API_KEY; writes data/judge_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROMPT_NAME = "lab11-judge"
DATASET_NAME = "lab11-judge-set"

# Mustache templates, exactly as they appear in Phoenix's Prompt Playground.
SYSTEM_V1 = "You are a senior engineer reviewing a pull request produced by an autonomous coding agent."

USER_V1 = """## Issue the agent was asked to fix
{{issue}}

## What the agent did (condensed transcript, most recent steps last)
{{transcript}}

## The agent's final patch
```diff
{{patch}}
```

Does this patch resolve the issue? Explain your reasoning in two or three sentences.
Then end with one final line that is exactly `VERDICT: resolved` or `VERDICT: not_resolved`."""

# The judge's verdict is its LAST "VERDICT" line. Case and markdown (e.g. "**Verdict:** resolved") are ignored.
VERDICT_RE = re.compile(r"VERDICT\W*(resolved|not_resolved)\b", re.I)
# The dataset evaluator (a built-in regex check in Phoenix) searches the judge's output for the TRUE verdict, using
# the same rule: "(?is)" = ignore case, "." matches newlines; the lookahead means "and no VERDICT after this one".
# "VERDICT\W*resolved" can't match "VERDICT: not_resolved", so a match means the judge was right.
TRUE_PATTERN = {True: r"(?is)VERDICT\W*resolved\b(?!.*VERDICT)",
                False: r"(?is)VERDICT\W*not_resolved\b(?!.*VERDICT)"}


def condense(trace: dict, max_steps: int = 25, max_output: int = 300) -> str:
    lines = []
    steps = [s for s in trace["steps"] if not s.get("format_error")]
    if len(steps) > max_steps:
        lines.append(f"[{len(steps) - max_steps} earlier steps omitted]")
        steps = steps[-max_steps:]
    for s in steps:
        if s["thought"]:
            lines.append(f"[{s['i']}] thought: {s['thought'][:300]}")
        for a in s["actions"]:
            out = (a["output"] or "").strip().replace("\n", " | ")[:max_output]
            lines.append(f"[{s['i']}] $ {a['command'][:300]}\n    -> rc={a['returncode']} {out}")
    lines.append(f"[exit status: {trace['exit_status']}]")
    return "\n".join(lines)


def judge_inputs(trace: dict) -> dict:
    """The three template variables, as stored in the Phoenix dataset."""
    return {"issue": trace["problem_statement"][:4000],
            "transcript": condense(trace),
            "patch": (trace["patch"] or "(empty patch)")[:6000]}


def parse_verdict(text: str) -> str | None:
    """'resolved', 'not_resolved', or None if the judge didn't follow the output format. Uses the LAST match."""
    found = VERDICT_RE.findall(text or "")
    return found[-1].lower() if found else None


def render(template: str, variables: dict) -> str:
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: str(variables[m.group(1)]), template)


def judge_trace(trace: dict, model: str, system: str = SYSTEM_V1, user: str = USER_V1) -> dict:
    import litellm  # imported lazily so the notebook works without it

    resp = litellm.completion(model=model, temperature=0, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": render(user, judge_inputs(trace))}])
    text = resp.choices[0].message.content or ""
    verdict = parse_verdict(text)
    return {"trace_id": trace["trace_id"], "judge_model": model,
            "judge_says_resolved": verdict == "resolved", "judge_reason": text.strip()[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", default="data/traces.jsonl")
    ap.add_argument("--out", default="data/judge_outputs.jsonl")
    args = ap.parse_args()
    traces = [json.loads(l) for l in Path(args.traces).read_text().splitlines() if l.strip()]
    with open(args.out, "w") as f:
        for i, t in enumerate(traces, 1):
            row = judge_trace(t, args.model)
            f.write(json.dumps(row) + "\n")
            print(f"{i}/{len(traces)} {t['trace_id']}: judge={row['judge_says_resolved']} truth={t['resolved']}")


if __name__ == "__main__":
    main()

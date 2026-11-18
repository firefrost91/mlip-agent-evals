"""Run mini-swe-agent on the toy repo task (optional: trace your own run with optional/tracing.py).

With GEMINI_API_KEY set (from the course's Google credits), the agent is driven by a real Gemini model. Without
it, a scripted model replays a realistic run through the exact same code path (LiteLLM's mock mode), so tracing
looks the same either way and nobody gets blocked on keys.

    from run_agent import run_agent
    info = run_agent()                     # one run on the toy task
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

TOY_REPO = Path(__file__).parent / "toy_repo"
TASK = ("Checkout totals are wrong whenever a discount is applied: a 10% discount on a $50.00 cart charges "
        "$0.00 instead of $45.00, and order_total() is affected too. Please fix it.")
DEFAULT_MODEL = os.environ.get("LAB_MODEL", "gemini/gemini-3.5-flash")

# What the scripted model "decides" at each step, used when there is no API key.
SCRIPT = [
    ("Let me look at the repository layout.", "find . -name '*.py' -not -path './.git/*' | sort"),
    ("Let me read the pricing module.", "cat shop/pricing.py"),
    ("Let me run the pricing tests.", "python -m pytest -q tests/test_prices.py"),
    ("That file doesn't exist. Let me run the whole test suite instead.", "python -m pytest -q tests/ 2>&1 | tail -5"),
    ("percent / 10 should be percent / 100. Let me fix it.",
     "python -c \"import pathlib; p = pathlib.Path('shop/pricing.py'); "
     "p.write_text(p.read_text().replace('percent / 10)', 'percent / 100)'))\""),
    ("Let me rerun the tests to verify the fix.", "python -m pytest -q tests/ 2>&1 | tail -3"),
    ("All tests pass. Let me create the patch.", "git diff -- shop/pricing.py > patch.txt && cat patch.txt"),
    ("Submitting.", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
]


class ScriptedModel(LitellmModel):
    """A real LitellmModel whose replies come from SCRIPT via LiteLLM's mock mode (no network, no key)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._i = 0

    def _query(self, messages, **kwargs):
        thought, cmd = SCRIPT[min(self._i, len(SCRIPT) - 1)]
        self._i += 1
        return super()._query(messages, mock_response=thought, mock_tool_calls=[{
            "id": f"call_{self._i}", "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
        }], **kwargs)


def _agent_config(step_limit: int) -> dict:
    cfg = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())["agent"]
    for key in ("system_template", "instance_template"):  # the SWE-bench prompt assumes a /testbed container
        cfg[key] = cfg[key].replace("/testbed", "the current working directory")
    cfg.update(step_limit=step_limit, cost_limit=1.0)
    return cfg


def run_agent(task: str = TASK, model: str | None = None, step_limit: int = 20, scripted: bool | None = None) -> dict:
    """Run the agent once on a fresh copy of the toy repo. Returns mini-swe-agent's info dict."""
    if scripted is None:
        scripted = not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    model_cls = ScriptedModel if scripted else LitellmModel
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(TOY_REPO, repo)
        subprocess.run("git init -q && git add . && git -c user.email=lab@cmu -c user.name=lab commit -qm init",
                       shell=True, cwd=repo, check=True)
        agent = DefaultAgent(model_cls(model_name=model or DEFAULT_MODEL), LocalEnvironment(cwd=str(repo)),
                             **_agent_config(step_limit))
        info = agent.run(task)
    print(f"[{'scripted' if scripted else model or DEFAULT_MODEL}] exit={info.get('exit_status')} "
          f"steps={agent.n_calls} cost=${agent.cost:.4f}")
    return info


if __name__ == "__main__":
    import tracing

    tracing.setup()
    tracing.add_agent_spans()
    run_agent()
    tracing.flush()

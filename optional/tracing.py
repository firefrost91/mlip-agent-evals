"""Tracing for mini-swe-agent: the instrumentation that produced the traces you debug in this lab.

setup() turns on OpenTelemetry and OpenInference's auto-instrumentation for LiteLLM, which every model call goes
through. That is what an off-the-shelf integration gives you: one span per LLM call, and nothing else.

add_agent_spans() wraps the agent's run loop, each step, and each shell command in spans, so a whole
run becomes one trace you can actually debug.
"""

from __future__ import annotations

from contextlib import contextmanager

import os

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

PHOENIX = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
_provider: TracerProvider | None = None


def setup(project: str = "lab11-my-agent") -> TracerProvider:
    """Send traces to Phoenix and auto-instrument LiteLLM. Call once, before running the agent."""
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"openinference.project.name": project}))
        _provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{PHOENIX}/v1/traces")))
        trace.set_tracer_provider(_provider)
        LiteLLMInstrumentor().instrument(tracer_provider=_provider)
    return _provider


def flush():
    if _provider is not None:
        _provider.force_flush()


def add_agent_spans():
    """Wrap DefaultAgent.run, DefaultAgent.step and LocalEnvironment.execute in spans."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.exceptions import InterruptAgentFlow

    if getattr(DefaultAgent, "_lab_traced", False):
        return
    tracer = trace.get_tracer("lab11.agent")
    orig_run, orig_step, orig_execute = DefaultAgent.run, DefaultAgent.step, LocalEnvironment.execute

    @contextmanager
    def agent_span(name):
        """A span that treats mini-swe-agent's control-flow exceptions as normal, but real crashes as errors.

        mini-swe-agent ends a run by raising InterruptAgentFlow (Submitted, LimitsExceeded, ...). By default OTel
        marks any span an exception passes through as an error, so we switch that off and decide ourselves."""
        with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
            try:
                yield span
            except InterruptAgentFlow as e:  # e.g. the submit command: control flow, not a failure
                span.set_attribute("agent.control_flow", type(e).__name__)
                raise
            except Exception as e:  # a real crash (API error, bug, ...): keep it visible
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"))
                raise

    def run(self, task="", **kwargs):
        with agent_span("agent.run") as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("input.value", task)
            result = orig_run(self, task, **kwargs)
            span.set_attribute("output.value", result.get("submission") or "(empty patch)")
            span.set_attribute("agent.exit_status", result.get("exit_status", ""))
            span.set_attribute("agent.n_steps", self.n_calls)
            span.set_attribute("agent.cost_usd", self.cost)
            return result

    def step(self):
        with agent_span(f"agent.step {self.n_calls}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("agent.step_index", self.n_calls)
            return orig_step(self)

    def execute(self, action, *args, **kwargs):
        command = action["command"] if isinstance(action, dict) else str(action)
        with agent_span("tool.bash") as span:
            span.set_attribute("openinference.span.kind", "TOOL")
            span.set_attribute("tool.name", "bash")
            span.set_attribute("input.value", command)
            output = orig_execute(self, action, *args, **kwargs)
            rc = output.get("returncode")
            span.set_attribute("output.value", output.get("output", ""))
            span.set_attribute("tool.returncode", rc if rc is not None else -1)
            if rc == 0:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, f"exit code {rc}"))
            return output

    DefaultAgent.run, DefaultAgent.step, LocalEnvironment.execute = run, step, execute
    DefaultAgent._lab_traced = True

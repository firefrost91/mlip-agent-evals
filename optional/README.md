# Optional: run the instrumented agent yourself

Everything in the lab works from the recorded traces. If you want to see the instrumentation produce a trace live:

```bash
phoenix serve                     # if it isn't already running
cd optional
python run_agent.py               # one run on toy_repo/, a small shop package with a discount bug
```

Then open the `lab11-my-agent` project in Phoenix. Without `GEMINI_API_KEY` a scripted model replays a realistic run
through the same code path; with it (course Google credits), a real Gemini model drives the agent. Set `LAB_MODEL`
to choose a different model.

`tracing.py` is the instrumentation: OpenInference's LiteLLM auto-instrumentation plus custom spans for the agent
run, each step, and each shell command.

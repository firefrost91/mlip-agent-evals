# Lab 11 starter code: Debugging Agents from Production Traces

Starter repository for Lab 11 of Machine Learning in Production (17-445/17-645/17-745). The lab instructions are in
the course repository's Lab 11 README; this repository holds the code and data you need.

## Setup

```bash
git clone https://github.com/firefrost91/mlip-lab11-agent-tracing.git
cd mlip-lab11-agent-tracing
python3.12 -m venv .venv && source .venv/bin/activate     # or a conda env with Python 3.12 (not 3.14)
pip install -r requirements.txt

export GEMINI_API_KEY=...                                  # for Part 5 (course Google credits)
phoenix serve                                              # leave running; open http://localhost:6006
```

In a second terminal (same environment), open the lab and run the first code cell, which loads the traces into
Phoenix (about a minute):

```bash
source .venv/bin/activate
jupyter lab lab.ipynb
```

## What's in here

| File | Purpose |
|---|---|
| `lab.ipynb` | The lab: TODOs and questions for Parts 1 to 5 |
| `load_traces.py` | Loads the recorded agent runs into Phoenix, and sets up Part 5's judge dataset, prompt and evaluator |
| `helpers.py` | Pulls spans, annotations, experiments and summary data out of Phoenix into pandas (no need to edit) |
| `judge.py` | The starter LLM judge prompt (also loaded into Phoenix) |
| `data/` | The agent runs and summary data; sources and schema in `data/README.md` |
| `optional/` | The instrumentation (`tracing.py`, read it for Q3) and a script to trace your own agent run on a toy task |

## Data

The traces are real, public runs of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with Gemini 3.5
Flash on [SWE-bench Verified](https://www.swebench.com/), published by the SWE-bench team in
[SWE-bench/experiments](https://github.com/SWE-bench/experiments). See `data/README.md` for details.
# mlip-agent-evals

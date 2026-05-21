# Autoresearch Workflow Guide

This guide explains the part that is easiest to misunderstand: the difference
between running the training script and starting the research agent.

## The Two Roles

There are two different things involved in this project:

1. **The training script**: `uv run train.py`
2. **The research agent**: Codex, Claude, or another coding agent following `program.md`

The training script runs one experiment. The research agent runs the experiment
loop.

## What `uv run train.py` Does

`uv run train.py` trains whatever model is currently defined in `train.py`,
prints metrics like `val_bpb`, and exits.

When you run:

```bash
uv run train.py > run.log 2>&1
```

the output is saved to `run.log` instead of being printed to the terminal.

That command does **not**:

- choose a new experiment idea
- edit `train.py`
- commit anything
- compare against previous runs
- update `results.tsv`
- keep or discard code changes
- start an autonomous loop

## What the Research Agent Does

The research agent is the loop around the script. It should:

1. Pick an experiment idea.
2. Edit `train.py`.
3. Commit the experiment.
4. Run `uv run train.py > run.log 2>&1`.
5. Read `run.log`.
6. Update `results.tsv`.
7. Keep the change if `val_bpb` improves.
8. Revert or discard the change if `val_bpb` gets worse.
9. Repeat until stopped.

In short:

```text
uv run train.py
  Runs the current model once.
  Produces metrics.
  Exits.

research agent
  Chooses experiment ideas.
  Edits train.py.
  Runs uv run train.py.
  Reads run.log.
  Updates results.tsv.
  Keeps or discards changes.
  Repeats.
```

## Analogy

```text
uv run train.py = run the race once
research agent = coach who changes the training plan, runs races, records scores, and keeps improving
```

## Manual Workflow

Use this when you are learning the project and want to run one experiment
yourself:

```bash
# Prepare the dataset and tokenizer once.
uv run prepare.py

# Run one experiment and save all output.
uv run train.py > run.log 2>&1

# Read the key metrics.
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:\|^total_seconds:" run.log
```

Then add a row to `results.tsv` yourself or have the agent do it.

The results file is a tab-separated scoreboard:

```text
commit	val_bpb	memory_gb	status	description
```

## Autonomous Agent Workflow

Use this when setup is complete and you want the agent to start improving the
model.

Say:

```text
Start the research loop. Try to improve val_bpb from the baseline, update results.tsv after each run, and keep going until I stop you.
```

That prompt is the instruction to the agent. You do not need to manually run
`uv run train.py > run.log 2>&1` again to start the autonomous loop. The agent
runs that command as one step inside the loop.

## Current Baseline

On the folktales branch, the current baseline is:

```text
val_bpb: 1.842572
```

Lower `val_bpb` is better. Future experiments should try to beat that number.

## Apple GPU / MPS Note

On Apple Silicon, this project uses PyTorch MPS when available.

Device selection happens in `train.py`:

```python
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

On a Mac without an NVIDIA GPU, CUDA is unavailable and MPS is available, so the
script chooses `mps`.

For this macOS/MPS fork, `peak_vram_mb` may show `0.0` even when the Apple GPU is
being used. The current code only measures CUDA peak memory with
`torch.cuda.max_memory_allocated()`.

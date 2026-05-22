# autoresearch-macos

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Open source project worth to look at

Open source collabaration platform for agentic swarms in organizations and communityies. 

[SentientWave Automata](https://github.com/sentientwave/automata)

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4 with Metal/MPS support) or a single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Learning docs

This branch includes two short guides for the concepts that are easiest to confuse:

- [Autoresearch Workflow Guide](docs/autoresearch-workflow.md) explains what the training script does, what the research agent does, how `run.log` and `results.tsv` work, and the exact prompt to start the autonomous loop.
- [Folktales Data Pipeline](docs/folktales-data-pipeline.md) explains how `prepare.py` downloads and prepares `merve/folk-mythology-tales`.
- [Folktales Training Run Explained](docs/experiment-run-explained.html) explains the first autoresearch run in plain English: what a model is, which parameters changed, why they mattered, and what each experiment taught us.

**Platforms support**. This fork officially supports **macOS (Apple Silicon / MPS)** and CPU environments, while preserving the original NVIDIA GPU support. It removes the hardcoded dependency on FlashAttention-3, falling back to PyTorch's native Scaled Dot Product Attention (SDPA) with manual sliding window causal masking when needed. It also features MPS-specific optimizations (disabling unsupported `torch.compile` paths, lowering memory batch sizes for Metal bounds, and precisely casting optimizer states) allowing you to run autonomous research agents directly on your Mac!

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

After setup is complete and the baseline is recorded, the next step is not for you to run the Python command again manually. The next step is to tell the coding agent to begin the autonomous research loop.

Use a prompt like:

```
Start the research loop. Try to improve val_bpb from the baseline, update results.tsv after each run, and keep going until I stop you.
```

At that point, the agent should handle the loop:

```
1. Pick an experiment idea.
2. Edit train.py.
3. Commit the experiment.
4. Run uv run train.py > run.log 2>&1.
5. Read run.log.
6. Update results.tsv.
7. Keep the change if val_bpb improves.
8. Revert or discard the change if val_bpb gets worse.
9. Repeat.
```

So there are two different "next steps" depending on who is driving:

```
Manual human workflow:
  You run uv run train.py > run.log 2>&1 yourself.
  Then you or the agent records the result.

Autonomous agent workflow:
  You tell the agent: Start the research loop.
  The agent runs uv run train.py > run.log 2>&1 as one step inside the loop.
```

See [Autoresearch Workflow Guide](docs/autoresearch-workflow.md) for the full mental model.

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — data preparation, tokenizer training, dataloader, fixed evaluation
train.py        — model, optimizer, hyperparameters, training loop
program.md      — instructions the research agent follows
results.tsv     — experiment scoreboard, updated after each completed run
run.log         — raw output from the most recent redirected training run, usually uncommitted
pyproject.toml  — Python dependencies
```

### What each file does

- **`prepare.py`** downloads and prepares data. On this branch it downloads `merve/folk-mythology-tales`, creates local train/validation parquet files, trains the tokenizer, and exposes the dataloader and validation metric. During research experiments this file should be treated as fixed.
- **`train.py`** is the experiment surface. The agent changes model size, architecture, optimizer settings, batch sizes, learning rates, and training-loop details here.
- **`program.md`** is the agent playbook. It tells the coding agent how to set up a branch, run experiments, record results, and decide whether to keep or discard a change.
- **`results.tsv`** is the scoreboard. `train.py` does not update it automatically; the agent or human records each experiment after reading `run.log`.
- **`run.log`** is the captured output from a training run. It is useful for debugging and metric extraction, but it is normally not committed.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

If you're going to be using autoresearch on Apple Macbooks in particular, I'd recommend one of the forks below. On top of this, if you'd like half-decent results at such a small scale, I'd recommend this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) which is cleaner than what exists out there otherwise. It should be a drop in replacement because I have encoded it in exactly the same format. Any of your favorite coding agents should be able to do the swap :)

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## License

MIT

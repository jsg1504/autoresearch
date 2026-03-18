# autoresearch

Autonomous Triton kernel optimization for the [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/), built on the [autoresearch](https://github.com/karpathy/autoresearch) framework by @karpathy.

The idea: give an AI agent a Triton GPU kernel and let it optimize autonomously overnight. It modifies the kernel, submits to Modal for benchmarking on an NVIDIA B200, checks if the speedup improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a faster kernel.

## Competition track

**`gdn_decode_qk4_v8_d128_k_last`** — Gated Delta Net single-token decode kernel from Qwen3-Next linear attention layers (GVA configuration, TP=4). The kernel performs a recurrent state update with a `[128, 128]` f32 state matrix per batch per v_head.

The metric is **speedup_factor** vs the reference implementation — higher is better. Correctness is a hard constraint: all workloads must pass.

## How it works

The repo adapts @karpathy's autoresearch autonomous experiment loop for GPU kernel optimization instead of LLM training:

- **`mlsys26/solution/triton/kernel.py`** — the single file the agent edits. Contains the Triton JIT kernel and Python entry point. Autotuning configs, parallelization strategy, memory access patterns — everything is fair game. **This file is edited and iterated on by the agent**.
- **`program.md`** — agent instructions for the autonomous optimization loop. **This file is edited and iterated on by the human**.
- **`mlsys26/`** — the [FlashInfer-Bench starter kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit) (submodule). Contains contest config, benchmark scripts, dataset definitions, and workloads. Not modified by the agent.

## Quick start

**Requirements:** Python 3.12+, [Modal](https://modal.com/) account, `flashinfer-bench`.

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/jsg1504/autoresearch.git
cd autoresearch

# 2. Set up Python environment
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal

# 3. Set up Modal (one-time)
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26/mlsys26-contest/

# 4. Verify setup
python -c "import flashinfer_bench; print('OK')"
modal volume ls flashinfer-trace

# 5. Run a single benchmark to establish baseline
modal run mlsys26/scripts/run_modal.py
```

## Running the agent

Spin up your AI coding agent (Claude, Codex, etc.) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The `program.md` file contains the full autonomous loop instructions — the agent reads it, creates a branch, establishes a baseline, then iterates on the kernel indefinitely.

## Project structure

```
program.md                              — agent instructions (human edits this)
program.ko.md                           — Korean translation of program.md
mlsys26/                                — FlashInfer-Bench starter kit (submodule)
  solution/triton/kernel.py             — Triton kernel (agent edits this)
  config.toml                           — contest track configuration
  scripts/run_modal.py                  — Modal B200 benchmark runner
  scripts/run_local.py                  — local benchmark runner
  scripts/pack_solution.py              — solution packer
  mlsys26-contest/                      — dataset (definitions + workloads)
```

## Design choices

- **Single file to modify.** The agent only touches `mlsys26/solution/triton/kernel.py`. This keeps the scope manageable and diffs reviewable.
- **Modal B200 benchmarking.** Every experiment runs on the actual competition target hardware (NVIDIA B200 via Modal). Each run takes ~1-3 minutes including cold start, giving ~20-40 experiments/hour.
- **Correctness as hard constraint.** Unlike LLM training where the metric is continuous (val_bpb), kernel optimization has a binary correctness gate — all workloads must produce numerically correct results. A fast but wrong kernel is immediately discarded.
- **Autonomous loop.** Same keep/discard pattern as the original autoresearch: if speedup improves and correctness passes, advance the branch. Otherwise, git reset and try something else.

## Credits

- Original autoresearch framework: [@karpathy](https://github.com/karpathy/autoresearch)
- FlashInfer-Bench: [flashinfer-ai/flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench)
- Contest: [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

## License

MIT

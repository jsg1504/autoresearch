# autoresearch

Autonomous Triton kernel optimization for the [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/).

Track: `gdn_decode_qk4_v8_d128_k_last` — Gated Delta Net single-token decode kernel, targeting NVIDIA Blackwell GPUs. The kernel implements a recurrent state update for Qwen3-Next linear attention layers (GVA configuration, TP=4).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The kernel optimization project lives in `mlsys26/`. Read these files for full context:
   - `mlsys26/solution/triton/kernel.py` — the file you modify. Contains the Triton JIT kernel and Python entry point.
   - `mlsys26/config.toml` — solution metadata and build config. Do not modify.
   - `mlsys26/mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json` — kernel definition with reference implementation, input/output shapes, and correctness constraints. Do not modify.
   - `mlsys26/scripts/run_modal.py` — Modal cloud benchmark runner (packs solution + runs benchmark on B200). Do not modify.
   - `mlsys26/scripts/pack_solution.py` — solution packer. Do not modify.
4. **Verify environment**: Check that:
   - `flashinfer-bench` is installed: `python -c "import flashinfer_bench; print('OK')"`
   - `modal` is installed and authenticated: `modal token list`
   - Modal volume `flashinfer-trace` exists with the dataset: `modal volume ls flashinfer-trace`
   - If Modal is not set up, tell the human to run the one-time setup:
     ```
     modal setup
     modal volume create flashinfer-trace
     modal volume put flashinfer-trace /path/to/mlsys26/mlsys26-contest/
     ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Kernel background

The kernel implements a **Gated Delta Net decode** step — a single-token recurrent state update used in Qwen3-Next's linear attention layers. Understanding the math is essential for optimization:

**Configuration** (GVA from Qwen3-Next, TP=4):
- num_q_heads = num_k_heads = 4, num_v_heads = 8, head_dim = 128
- State layout: k-last `[B, H_v, V, K]` where V=K=128
- Variable batch_size, seq_len always 1 (decode)

**Mathematical operation** (per batch, per v_head):
```
g      = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))    # decay gate
beta   = sigmoid(b[h])                                         # update gate
S_dec  = g * state                                             # decay state
old_v  = k_exp . S_dec                                         # retrieve (mat-vec)
delta  = beta * (v - old_v)                                    # gated prediction error
S_new  = S_dec + outer(k_exp, delta)                           # rank-1 state update
output = scale * q_exp . S_new                                 # query output (mat-vec)
```

**Key insight**: The bottleneck is the state tensor `[128, 128]` per batch per v_head — a 64KB f32 block. The kernel's performance is dominated by how efficiently it reads, transforms, and writes this state.

## Experimentation

Each experiment submits the kernel to Modal for benchmarking on an NVIDIA B200 GPU (the actual competition target hardware). You launch it as:

```
modal run mlsys26/scripts/run_modal.py > run.log 2>&1
```

This packs `mlsys26/solution/triton/kernel.py` into `solution.json` locally, then sends it to a Modal B200 instance for benchmarking against all workloads.

**What you CAN do:**
- Modify `mlsys26/solution/triton/kernel.py` — this is the only file you edit. Everything is fair game: autotuning configurations (BLOCK_V, num_warps, num_stages), parallelization strategy (grid decomposition, tiling), memory access patterns, loop ordering, algorithmic restructuring, numerical tricks, shared memory usage, register optimization, etc. Both the `_gdn_decode_kernel` JIT function and the `kernel()` Python wrapper are fair game.

**What you CANNOT do:**
- Modify any files outside `mlsys26/solution/triton/kernel.py`. The scripts, config, definitions, and benchmark framework are read-only.
- Change the `kernel()` function signature. The entry point `kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)` must accept exactly these parameters with the documented shapes and dtypes (Destination Passing Style).
- Break correctness. The kernel must produce numerically correct results across ALL workloads. A faster but incorrect kernel is worthless.
- Install new packages or add dependencies.

**The goal is simple: maximize average speedup_factor across all workloads while maintaining correctness.** The speedup_factor is your kernel's latency vs the reference implementation — higher is better. A speedup of 1.0x means matching the reference; >1.0x means faster.

**Correctness** is a hard constraint. If ANY workload fails correctness (status != "passed"), the entire experiment is invalid regardless of speedup. This is non-negotiable.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the benchmark with the current kernel as-is.

## Output format

The benchmark prints results per workload like this:

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: passed | 0.042 ms | 1.85x speedup | abs_err=3.05e-05, rel_err=1.92e-03
  Workload d66ae544...: passed | 0.041 ms | 1.90x speedup | abs_err=2.89e-05, rel_err=1.75e-03
  ...
```

Each line shows: status, latency, speedup vs reference, and correctness metrics.

You can extract results from the log:

```
# Extract all speedup values
grep "speedup" run.log

# Compute average and minimum speedup
grep "speedup" run.log | grep -oP '[\d.]+(?=x speedup)' | awk '{s+=$1; if(NR==1||$1<m)m=$1; n++} END{printf "avg=%.6f min=%.6f n=%d\n",s/n,m,n}'

# Check for correctness failures (should produce no output if all passed)
grep "Workload" run.log | grep -v "passed"

# If grep output is empty or shows "Error"/"Traceback", the run crashed
grep -c "speedup" run.log
```

Note: Modal output may include additional spinup/log lines beyond the benchmark results — the grep commands above filter for the relevant data. Triton compiles kernels on first run with new autotune configs; this compilation happens on the remote B200 and is handled by the benchmark's warmup.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	avg_speedup	min_speedup	latency_ms	status	description
```

1. git commit hash (short, 7 chars)
2. average speedup_factor across all workloads (e.g. 1.850000) — use 0.000000 for crashes
3. minimum speedup_factor across all workloads (e.g. 1.750000) — use 0.000000 for crashes
4. average latency in ms, round to .3f (e.g. 0.042) — use 0.000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	avg_speedup	min_speedup	latency_ms	status	description
a1b2c3d	1.850000	1.750000	0.042	keep	baseline
b2c3d4e	2.100000	1.900000	0.038	keep	BLOCK_V=16 with num_warps=4
c3d4e5f	1.600000	1.400000	0.045	discard	num_stages=2 pipelining (slower)
d4e5f6g	0.000000	0.000000	0.000	crash	2D tiling experiment (correctness failed)
e5f6g7h	0.000000	0.000000	0.000	crash	shared memory approach (compilation error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `mlsys26/solution/triton/kernel.py` with an experimental optimization idea by directly hacking the code.
3. git commit
4. Run the experiment: `modal run mlsys26/scripts/run_modal.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "speedup" run.log` and compute avg/min with the awk command from the output format section.
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. **Correctness gate**: Run `grep "Workload" run.log | grep -v "passed"`. If this produces ANY output, correctness failed — treat as crash.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If avg_speedup improved (higher) AND all workloads passed, you "advance" the branch, keeping the git commit
10. If avg_speedup is equal or worse, or correctness failed, you git reset back to where you started

The idea is that you are a completely autonomous kernel optimization researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck, try more radical approaches.

**Timeout**: Each Modal benchmark typically completes in 1-3 minutes (includes cold start + compilation + benchmark). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (compilation error, correctness failure, runtime error, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a syntax error, a missing import), fix it and re-run. If the idea itself is fundamentally broken (e.g. the approach can't maintain correctness), just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — study the Triton programming model, analyze the kernel's memory access patterns, try different parallelization strategies, experiment with tiling approaches, try different autotuning configurations, combine previous near-misses. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each Modal benchmark takes ~1-3 minutes (including cold start overhead), so you can run approx 20-40 experiments/hour (depending on Modal spinup times and how quickly you iterate on ideas), for a total of hundreds of experiments overnight. The user then wakes up to a log of optimization experiments and (hopefully) a faster kernel. The key advantage of using Modal is that benchmarks run on actual B200 hardware — the same GPU used for competition evaluation.

## Optimization ideas to explore

Here are directions for kernel optimization. Not exhaustive — be creative.

### Autotuning parameters
- **BLOCK_V**: try 2, 4, 8, 16, 32, 64, 128 (how many V-rows each program handles)
- **num_warps**: 1, 2, 4, 8 (concurrent warp groups per program)
- **num_stages**: 1, 2, 3, 4 (software pipelining stages for async data movement)
- **Autotune key**: currently only `["B"]` — consider adding other dimensions if relevant
- **Config diversity**: more combinations of (BLOCK_V, num_warps, num_stages) in the autotune list
- Think about which configs work best for small batch (B=1) vs larger batch sizes

### Parallelization strategy
- Current: 1D grid over (batch × v_head × v_block), each program handles BLOCK_V rows of V and all 128 K columns
- Try 2D tiling: split both V and K dimensions into blocks (K-blocking with accumulation)
- Try different grid decompositions — what if each program handles all V for one (batch, v_head)?
- Consider processing multiple v_heads per program for small batch sizes

### Memory access patterns
- **State is the bottleneck**: `[128, 128]` f32 per (batch, v_head) = 64KB. Optimizing state access is the #1 lever
- Ensure coalesced loads along the K dimension (innermost, already contiguous)
- K-blocking: instead of loading all 128 K columns at once, process in K-chunks to reduce register pressure
- Explore `tl.load` with `eviction_policy` hints (`"evict_first"`, `"evict_last"`)
- Try vectorized loads (loading wider elements)
- Consider whether transposing the state or reordering accesses could help

### Compute optimization
- Use `tl.dot` for the mat-vec operations (old_v and output computation) if BLOCK_V and K dimensions are suitable
- Fuse operations to minimize register traffic: combine state decay + retrieve in one pass
- Pre-compute shared intermediates (g, beta are scalar per v_head — already efficient)
- Consider whether combining the two mat-vec operations (old_v retrieval and output query) into a single state traversal helps
- Look for opportunities to reduce the number of state reads/writes

### Numerical precision
- Gate computation already has a stable softplus branch — explore tighter thresholds
- Where is f32 strictly needed vs where bf16 might suffice? (careful — correctness constraint!)
- State is f32 by spec; don't try to change it. But intermediate computations might allow mixed precision

### Hardware-aware optimization
- Target B200 (148 SMs, 8 TB/s HBM3e) — grid size should fill or slightly overfill SMs
- For batch_size=1 with 8 v_heads: 8 × (128/BLOCK_V) programs. At BLOCK_V=4 → 256 programs (good fill)
- This kernel is likely memory-bound — optimize for memory bandwidth, not compute throughput
- Register pressure vs occupancy: more registers per thread → fewer concurrent threads → potentially better register reuse but lower occupancy
- Consider L2 cache effects for state access patterns

# autoresearch

[FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)을 위한 자율 Triton kernel 최적화.

Track: `gdn_decode_qk4_v8_d128_k_last` — Gated Delta Net single-token decode kernel, NVIDIA Blackwell GPU 대상. Qwen3-Next linear attention layer의 recurrent state update를 구현하는 kernel (GVA configuration, TP=4).

## Setup

새 실험을 시작하기 위해 사용자와 함께 다음을 진행:

1. **run tag 합의**: 오늘 날짜 기반으로 tag 제안 (예: `mar19`). branch `autoresearch/<tag>`가 이미 존재하면 안 됨 — 새로운 실행이어야 함.
2. **branch 생성**: 현재 master에서 `git checkout -b autoresearch/<tag>`.
3. **관련 파일 읽기**: kernel 최적화 프로젝트는 `mlsys26/`에 있음. 전체 맥락 파악을 위해 다음 파일들을 읽을 것:
   - `mlsys26/solution/triton/kernel.py` — 수정할 파일. Triton JIT kernel과 Python entry point 포함.
   - `mlsys26/config.toml` — solution metadata와 build config. 수정 불가.
   - `mlsys26/mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json` — kernel definition (reference implementation, input/output shape, correctness constraint 포함). 수정 불가.
   - `mlsys26/scripts/run_modal.py` — Modal cloud benchmark runner (solution pack + B200에서 benchmark 실행). 수정 불가.
   - `mlsys26/scripts/pack_solution.py` — solution packer. 수정 불가.
4. **환경 확인**: 다음을 체크:
   - `flashinfer-bench` 설치 여부: `python -c "import flashinfer_bench; print('OK')"`
   - `modal` 설치 및 인증 상태: `modal token list`
   - Modal volume `flashinfer-trace`에 dataset이 있는지 확인: `modal volume ls flashinfer-trace`
   - Modal이 설정되어 있지 않으면 사용자에게 일회성 setup 실행을 요청:
     ```
     modal setup
     modal volume create flashinfer-trace
     modal volume put flashinfer-trace /path/to/mlsys26/mlsys26-contest/
     ```
5. **results.tsv 초기화**: header row만 있는 `results.tsv` 생성. baseline은 첫 번째 실행 후 기록.
6. **확인 후 시작**: setup이 정상인지 확인.

확인을 받으면 실험을 시작.

## Kernel 배경

이 kernel은 **Gated Delta Net decode** step을 구현함 — Qwen3-Next의 linear attention layer에서 사용하는 single-token recurrent state update. 최적화를 위해 수학적 이해가 필수:

**Configuration** (Qwen3-Next의 GVA, TP=4):
- num_q_heads = num_k_heads = 4, num_v_heads = 8, head_dim = 128
- State layout: k-last `[B, H_v, V, K]` (V=K=128)
- batch_size는 가변, seq_len은 항상 1 (decode)

**수학적 연산** (batch당, v_head당):
```
g      = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))    # decay gate
beta   = sigmoid(b[h])                                         # update gate
S_dec  = g * state                                             # state decay
old_v  = k_exp . S_dec                                         # 검색 (mat-vec)
delta  = beta * (v - old_v)                                    # gated prediction error
S_new  = S_dec + outer(k_exp, delta)                           # rank-1 state update
output = scale * q_exp . S_new                                 # query output (mat-vec)
```

**핵심 포인트**: 병목은 state tensor `[128, 128]` (batch당, v_head당 64KB f32 block). kernel 성능은 이 state를 얼마나 효율적으로 읽고, 변환하고, 쓰느냐에 좌우됨.

## 실험

각 실험은 kernel을 Modal에 제출하여 NVIDIA B200 GPU(실제 대회 평가 하드웨어)에서 benchmark를 실행. 다음과 같이 실행:

```
modal run mlsys26/scripts/run_modal.py > run.log 2>&1
```

이 명령은 로컬에서 `mlsys26/solution/triton/kernel.py`를 `solution.json`으로 pack한 후, Modal B200 instance로 전송하여 모든 workload에 대해 benchmark를 실행.

**할 수 있는 것:**
- `mlsys26/solution/triton/kernel.py` 수정 — 이 파일만 편집. autotuning configuration (BLOCK_V, num_warps, num_stages), parallelization strategy (grid decomposition, tiling), memory access pattern, loop ordering, 알고리즘 재구성, numerical trick, shared memory 사용, register 최적화 등 모든 것이 가능. `_gdn_decode_kernel` JIT function과 `kernel()` Python wrapper 모두 수정 가능.

**할 수 없는 것:**
- `mlsys26/solution/triton/kernel.py` 외의 파일 수정 불가. script, config, definition, benchmark framework는 read-only.
- `kernel()` function signature 변경 불가. entry point `kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)`는 문서화된 shape과 dtype을 그대로 받아야 함 (Destination Passing Style).
- correctness 훼손 불가. kernel은 모든 workload에서 수치적으로 정확한 결과를 생성해야 함. 빠르지만 부정확한 kernel은 무가치.
- 새 package 설치나 dependency 추가 불가.

**목표는 단순함: correctness를 유지하면서 모든 workload의 평균 speedup_factor를 최대화.** speedup_factor는 reference implementation 대비 kernel의 latency — 높을수록 좋음. speedup 1.0x는 reference와 동일, >1.0x는 더 빠른 것.

**Correctness**는 hard constraint. 어떤 workload이든 correctness에 실패하면 (status != "passed") speedup에 관계없이 해당 실험 전체가 무효. 이것은 협상 불가.

**단순성 기준**: 다른 조건이 같다면 단순한 쪽이 좋음. 못생긴 복잡성을 추가하면서 얻는 작은 개선은 가치가 없음. 반대로, 무언가를 제거하고도 같거나 더 나은 결과를 얻는 것은 훌륭한 성과 — 단순화 승리. 변경 사항을 유지할지 판단할 때, 복잡성 비용 대비 개선 규모를 저울질할 것.

**첫 번째 실행**: 첫 실행은 항상 baseline 확립을 위해, 현재 kernel 그대로 benchmark를 실행.

## Output format

benchmark는 workload별 결과를 다음과 같이 출력:

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: passed | 0.042 ms | 1.85x speedup | abs_err=3.05e-05, rel_err=1.92e-03
  Workload d66ae544...: passed | 0.041 ms | 1.90x speedup | abs_err=2.89e-05, rel_err=1.75e-03
  ...
```

각 줄에는: status, latency, reference 대비 speedup, correctness metric이 표시.

log에서 결과를 추출하는 방법:

```
# 모든 speedup 값 추출
grep "speedup" run.log

# 평균 및 최소 speedup 계산
grep "speedup" run.log | grep -oP '[\d.]+(?=x speedup)' | awk '{s+=$1; if(NR==1||$1<m)m=$1; n++} END{printf "avg=%.6f min=%.6f n=%d\n",s/n,m,n}'

# correctness 실패 확인 (모두 통과하면 출력 없어야 함)
grep "Workload" run.log | grep -v "passed"

# grep 출력이 비어있거나 "Error"/"Traceback"이 보이면 crash한 것
grep -c "speedup" run.log
```

참고: Modal 출력에는 benchmark 결과 외에 추가적인 spinup/log 줄이 포함될 수 있음 — 위의 grep 명령이 관련 데이터만 필터링함. Triton은 새로운 autotune config으로 처음 실행 시 kernel을 compile하며, 이 compilation은 원격 B200에서 발생하고 benchmark의 warmup이 처리함.

## 결과 기록

실험이 끝나면 `results.tsv`에 기록 (tab-separated, 쉼표 사용 금지 — description에서 쉼표가 깨짐).

TSV는 header row와 6개 column으로 구성:

```
commit	avg_speedup	min_speedup	latency_ms	status	description
```

1. git commit hash (short, 7자)
2. 모든 workload의 평균 speedup_factor (예: 1.850000) — crash 시 0.000000
3. 모든 workload의 최소 speedup_factor (예: 1.750000) — crash 시 0.000000
4. 평균 latency (ms), .3f로 반올림 (예: 0.042) — crash 시 0.000
5. status: `keep`, `discard`, 또는 `crash`
6. 이 실험에서 시도한 내용에 대한 간단한 설명

예시:

```
commit	avg_speedup	min_speedup	latency_ms	status	description
a1b2c3d	1.850000	1.750000	0.042	keep	baseline
b2c3d4e	2.100000	1.900000	0.038	keep	BLOCK_V=16 with num_warps=4
c3d4e5f	1.600000	1.400000	0.045	discard	num_stages=2 pipelining (slower)
d4e5f6g	0.000000	0.000000	0.000	crash	2D tiling experiment (correctness failed)
e5f6g7h	0.000000	0.000000	0.000	crash	shared memory approach (compilation error)
```

## 실험 루프

실험은 전용 branch에서 진행 (예: `autoresearch/mar19`).

무한 반복:

1. git 상태 확인: 현재 branch/commit 확인
2. `mlsys26/solution/triton/kernel.py`를 실험적 최적화 아이디어로 직접 수정.
3. git commit
4. 실험 실행: `modal run mlsys26/scripts/run_modal.py > run.log 2>&1` (모든 출력을 redirect — tee를 사용하거나 출력이 context를 채우지 않도록 할 것)
5. 결과 확인: `grep "speedup" run.log`으로 추출하고 위 output format 섹션의 awk 명령으로 avg/min 계산.
6. grep 출력이 비어있으면 crash한 것. `tail -n 50 run.log`으로 Python stack trace를 읽고 수정 시도. 몇 번 시도해도 안 되면 포기.
7. **Correctness gate**: `grep "Workload" run.log | grep -v "passed"` 실행. 출력이 하나라도 있으면 correctness 실패 — crash로 처리.
8. 결과를 tsv에 기록 (주의: results.tsv 파일은 commit하지 말 것, git untracked 상태로 유지)
9. avg_speedup이 개선되고(높아지고) 모든 workload가 통과하면 branch를 "advance"하고 git commit 유지
10. avg_speedup이 같거나 나쁘거나 correctness가 실패하면 시작 지점으로 git reset

당신은 완전히 자율적인 kernel 최적화 연구자로서 다양한 시도를 함. 효과가 있으면 keep, 없으면 discard. branch를 advance하면서 반복. 막히면 더 과감한 접근을 시도.

**Timeout**: 각 Modal benchmark는 보통 1-3분 내에 완료됨 (cold start + compilation + benchmark 포함). 실행이 10분을 초과하면 kill하고 실패로 처리 (discard 후 revert).

**Crash**: 실행이 crash하면 (compilation error, correctness failure, runtime error 등) 판단을 사용: 단순한 문제(예: 오타, syntax error, missing import)라면 수정 후 재실행. 아이디어 자체가 근본적으로 문제(예: correctness를 유지할 수 없는 접근)라면 건너뛰고 tsv에 "crash" status로 기록하고 다음으로 진행.

**절대 멈추지 말 것**: 실험 루프가 시작되면 (초기 setup 이후) 사용자에게 계속할지 물어보지 말 것. "계속 할까요?" 또는 "여기서 멈출까요?"라고 묻지 말 것. 사용자가 잠들어 있거나 컴퓨터를 떠났을 수 있으며, 당신이 수동으로 중지될 때까지 *무한히* 계속 작업하기를 기대함. 당신은 자율적. 아이디어가 고갈되면 더 열심히 생각할 것 — Triton programming model을 연구하고, kernel의 memory access pattern을 분석하고, 다른 parallelization strategy를 시도하고, tiling 접근법을 실험하고, 다른 autotuning configuration을 시도하고, 이전에 아깝게 실패한 것들을 조합. 루프는 사용자가 중단할 때까지 돌아감, 그게 전부.

사용 예시: 사용자가 잠자는 동안 실행해둘 수 있음. 각 Modal benchmark는 ~1-3분 소요(cold start overhead 포함)이므로, 시간당 약 20-40개의 실험을 실행할 수 있고 (Modal spinup 시간과 아이디어 반복 속도에 따라), 밤새 수백 개의 실험이 가능. 사용자는 아침에 일어나서 최적화 실험 로그와 (희망적으로) 더 빨라진 kernel을 확인. Modal 사용의 핵심 이점은 실제 B200 하드웨어에서 benchmark가 실행된다는 것 — 대회 평가에 사용되는 동일한 GPU.

## 탐색할 최적화 아이디어

kernel 최적화를 위한 방향들. 이것이 전부가 아님 — 창의적으로 접근할 것.

### Autotuning parameter
- **BLOCK_V**: 2, 4, 8, 16, 32, 64, 128 시도 (각 program이 처리하는 V-row 수)
- **num_warps**: 1, 2, 4, 8 (program당 concurrent warp group 수)
- **num_stages**: 1, 2, 3, 4 (async data movement를 위한 software pipelining stage)
- **Autotune key**: 현재 `["B"]`만 사용 — 다른 dimension 추가 고려
- **Config 다양성**: autotune list에 (BLOCK_V, num_warps, num_stages)의 더 많은 조합
- small batch (B=1)와 큰 batch size에서 어떤 config이 최적인지 고려

### Parallelization strategy
- 현재: (batch × v_head × v_block)에 대한 1D grid, 각 program이 BLOCK_V개의 V row와 128개의 K column 전체를 처리
- 2D tiling 시도: V와 K dimension을 모두 block으로 분할 (K-blocking with accumulation)
- 다른 grid decomposition 시도 — 각 program이 하나의 (batch, v_head)에 대해 V 전체를 처리하면?
- small batch size에서 여러 v_head를 하나의 program이 처리하는 것을 고려

### Memory access pattern
- **State가 병목**: batch당, v_head당 `[128, 128]` f32 = 64KB. state access 최적화가 가장 큰 레버
- K dimension(innermost, 이미 contiguous)을 따른 coalesced load 보장
- K-blocking: 128개 K column을 한 번에 load하는 대신, K-chunk 단위로 처리하여 register pressure 감소
- `tl.load`에 `eviction_policy` hint (`"evict_first"`, `"evict_last"`) 탐색
- vectorized load 시도 (더 넓은 element loading)
- state를 transpose하거나 access 순서를 바꾸는 것이 도움이 되는지 검토

### Compute 최적화
- BLOCK_V와 K dimension이 적합하면 mat-vec 연산 (old_v, output computation)에 `tl.dot` 사용
- register traffic 최소화를 위한 연산 fusion: state decay + retrieve를 한 pass로 결합
- 공유 중간값 pre-compute (g, beta는 v_head당 scalar — 이미 효율적)
- 두 mat-vec 연산 (old_v retrieve와 output query)을 하나의 state traversal로 결합하는 것이 도움되는지 검토
- state read/write 횟수를 줄일 기회 탐색

### Numerical precision
- gate computation에 이미 안정적인 softplus branch가 있음 — 더 tight한 threshold 탐색
- f32가 반드시 필요한 곳 vs bf16이 가능한 곳은? (주의 — correctness constraint!)
- State는 spec상 f32; 변경하지 말 것. 그러나 중간 연산은 mixed precision 가능할 수 있음

### Hardware-aware 최적화
- B200 대상 (148 SM, 8 TB/s HBM3e) — grid size가 SM을 꽉 채우거나 약간 초과해야 함
- batch_size=1, 8 v_head인 경우: 8 × (128/BLOCK_V) program. BLOCK_V=4이면 → 256 program (좋은 fill)
- 이 kernel은 memory-bound일 가능성이 높음 — compute throughput이 아닌 memory bandwidth에 최적화
- Register pressure vs occupancy: thread당 register가 많으면 → concurrent thread 감소 → register 재사용은 좋지만 occupancy는 낮아짐
- state access pattern에 대한 L2 cache 효과 고려

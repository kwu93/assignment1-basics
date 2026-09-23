# Assignment 1 Experiment Log

All runs are tracked in wandb project `cs336-basics`.
Each run also writes `config.json` and `metrics.jsonl` under `checkpoints/<run_name>/`.

## Setup

- Hardware: Modal B200 (one GPU per run) via `modal_train.py`; local smoke tests on Apple M-series MPS
- dtype: bfloat16 on GPU, float32 locally
- Tokenizer / vocab: own BPE trained on TinyStories, vocab 10000 (`data/bpe`)
- Base config (TinyStories): vocab 10000, ctx 256, 4 layers, d_model 512, 16 heads, d_ff 1344, RoPE theta 10000
- Eval protocol: every 250 steps on random validation batches of 128 x 256 tokens, plus one eval after the final step.
  First screening pass used 20 batches (12% of the 5.46M-token validation set); everything after uses 50 batches (30%).
  Eval-to-eval jitter at 50 batches is about 0.01, so differences below that are not meaningful.
  The writeup number for any final model should be re-measured on the full validation set (166 batches).
- Token budget: 327,680,000 for full runs (batch 128 x 10,000 steps x 256), 81,920,000 for quarter-budget screening runs (2,500 steps).
- Schedule: cosine with warmup 3% of steps, decaying to lrmax / 10 (train.py defaults, commit 802947b).
- AdamW: betas (0.9, 0.95), weight decay 0.01, eps 1e-8, grad clip 1.0. Seed 42 for every run.

## Run naming

`<problem>_<variable>=<value>[_<suffix>]`, for example `lr_3e-3`, `bs_64`, `ablate_nonorm`.
Runs belonging to one sweep share a wandb group named after the problem.

## Entry format

Each experiment entry has the same shape.

```
### <run name>  (<date>)
Hypothesis:
Command:
Result:        final val loss X at step S, wall-clock T
Observation:
Next:
```

Write the hypothesis before the run starts.
Diverged and failed runs stay in the log.

---

## 7.2.2 learning_rate_tuning

Summary

| run | lr max | final val loss | steps | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|
| screen_lr_1e-4 | 1e-4 | 3.002 | 2500 | 157 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_3e-4 | 3e-4 | 2.171 | 2500 | 156 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_1e-3 | 1e-3 | 1.641 | 2500 | 155 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_2e-3 | 2e-3 | 1.577 | 2500 | 159 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget, 50 eval batches |
| screen_lr_3e-3 | 3e-3 | 1.630 | 2500 | 158 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_5e-3 | 5e-3 | 1.867 | 2500 | 173 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget, 50 eval batches; stable but falling behind |
| screen_lr_1e-2 | 1e-2 | 3.676 | 2500 | 158 s | [run](https://wandb.ai/porcini-labs/cs336-basics) | quarter budget; diverged after step 250, peak 4.44 at 1250 |
| lr_1e-3 | 1e-3 | 1.422 | 10000 | 598 s | [1r3q6wna](https://wandb.ai/porcini-labs/cs336-basics/runs/1r3q6wna) | full budget; under 1.45 from step 7000 |
| lr_2e-3 | 2e-3 | **1.389** | 10000 | 627 s | [l6c01mkt](https://wandb.ai/porcini-labs/cs336-basics/runs/l6c01mkt) | full budget; under 1.45 from step 6000; **base model** |
| lr_3e-3 | 3e-3 | 1.502 | 10000 | 635 s | [7bpj7o2f](https://wandb.ai/porcini-labs/cs336-basics/runs/7bpj7o2f) | full budget; never reaches 1.45 |

Search strategy: half-decade log grid over two decades around the Kingma et al. default of 1e-3 at a quarter budget, refine at a factor of 1.5 to 2 around the winner, then run the winner and its two neighbors at full budget.

Validation loss by step, full-budget runs:

| step | 1e-3 | 2e-3 | 3e-3 |
|---|---|---|---|
| 2000 | 1.702 | 1.694 | 1.782 |
| 4000 | 1.556 | 1.546 | 1.667 |
| 6000 | 1.463 | 1.446 | 1.567 |
| 8000 | 1.428 | 1.400 | 1.516 |
| 10000 | 1.422 | 1.389 | 1.502 |

Entries

### screen_lr_{1e-4,3e-4,1e-3,3e-3,1e-2}  (2026-09-21)
Hypothesis:  The Kingma default of 1e-3 is in the right range, and a two-decade log grid around it brackets the optimum on both sides.
Command:     GPU=B200 modal run modal_train.py --prefix screen_ --lrs 1e-4,3e-4,1e-3,3e-3,1e-2 --extra "--train-iters 2500 --batch-size 128 --dtype bfloat16 --eval-interval 250"
Result:      val loss at step 2250: 3.002 / 2.171 / 1.641 / 1.630 / 3.676; about 157 s each on B200; commit 802947b.
Observation: 1e-2 diverges after step 250 and peaks at 4.44 around step 1250 before partially recovering.
             1e-4 and 3e-4 are far too slow and still improving.
             1e-3 and 3e-3 are tied within eval noise, with 1e-3 ahead for most of the run and 3e-3 ahead only at the last eval.
             No eval ran after the final step in this pass, so the last number is from 250 steps before the end.
Next:        Refine at 2e-3 and 5e-3 at quarter budget.
             Add a final eval after the last step, and raise eval batches from 20 to 50.

### screen_lr_{2e-3,5e-3}  (2026-09-21)
Hypothesis:  The optimum lies between 1e-3 and 3e-3, or between 3e-3 and the 1e-2 cliff.
Command:     GPU=B200 modal run modal_train.py --prefix screen_ --lrs 2e-3,5e-3 --extra "--train-iters 2500 --batch-size 128 --dtype bfloat16 --eval-interval 250 --eval-iters 50"
Result:      val loss at step 2500: 2e-3 = 1.577, 5e-3 = 1.867; about 165 s each on B200.
Observation: 2e-3 leads both neighbors at every eval from step 250 on, by about 0.06 at step 2250, well above eval noise.
             5e-3 never diverges but falls further behind as training goes on.
             The rate curve is peaked near 2e-3 and steeper on the high side.
Next:        Full-budget runs at 1e-3, 2e-3, 3e-3.

### lr_{1e-3,2e-3,3e-3}  (2026-09-21)
Hypothesis:  The quarter-budget optimum of 2e-3 holds at full budget, and the neighbors show which way it shifts if it moves.
Command:     GPU=B200 modal run --detach modal_train.py --lrs 1e-3,2e-3,3e-3 --extra "--train-iters 10000 --batch-size 128 --dtype bfloat16 --eval-interval 250 --eval-iters 50"
Result:      final val loss 1e-3 = 1.422, 2e-3 = 1.389, 3e-3 = 1.502; about 10.5 min each on B200; commit 802947b.
Observation: The ordering from the screen held at every eval, and stretching the cosine four times longer did not move the optimum.
             3e-3 degrades steadily and never reaches 1.45.
             2e-3 crosses 1.45 at step 6000 and 1e-3 at step 7000.
             The 1e-3 run's minimum came at step 9250 rather than the end, so eval noise at 50 batches is still about 0.01.
             The first attempt at these runs was killed at about step 8000 by a local client timeout; those wandb runs were deleted afterward.
             The partial curves matched the rerun; the rerun was from scratch to keep the wall-clock curves clean.
Next:        2e-3 is the base configuration for everything after this.
             Re-measure the lr_2e-3 checkpoint on the full validation set before quoting 1.389 in the writeup.
             Batch size sweep next, then the ablations, all at lr 2e-3.

## 7.2.2 batch_size_experiment

Summary

| run | batch size | lr max | final val loss | steps | tokens seen | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Entries

### 

## 7.2.3 generate

Checkpoint used:
Decoding settings (temperature, top-p, max tokens):

Samples

```
```

Commentary on fluency and failure modes:

## 7.3.1 layer_norm_ablation

Summary

| run | RMSNorm | final val loss | diverged? | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|
| | | | | | | |

Entries

### 

## 7.3.2 pre_norm_ablation

Summary

| run | norm placement | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|
| | | | | | |

Entries

### 

## 7.3.3 no_pos_emb

Summary

| run | positional encoding | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|
| | | | | | |

Entries

### 

## 7.3.4 swiglu_ablation

Summary

| run | FFN | d_ff | params | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Entries

### 

## 7.4 main_experiment (OpenWebText)

Summary

| run | config | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|
| | | | | | |

Entries

### 

## 7.5 leaderboard

Time limit: 90 minutes on H100.

| run | what changed | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|
| | | | | | |

Entries

### 

---

## Scratch and dead ends

Things tried that did not make it into a problem section: bugs found, throughput fixes, config mistakes.

- 2026-09-21: First bfloat16 run on Modal crashed in attention with "expected scalar type Float but found BFloat16".
  RoPE's cos and sin buffers were float32 and promoted Q and K while V stayed bfloat16.
  Fixed by casting the tables to the input dtype at use (commit 802947b).
- 2026-09-21: Old schedule defaults (lrmax 1, Tw 7, Tc 21) were unit-test leftovers and diverged within ten steps.
  Replaced with lrmax 1e-3 and warmup, cosine length, and lrmin derived from train-iters (commit 802947b).
- 2026-09-21: Per-step time at batch 128 bf16: 79 ms on H100, 54 ms on B200.
  A full run costs about $0.90 on either card, so B200 is the default for speed.
  The B200 speedup is smaller than the hardware gap, so the implementation is not saturating the GPU yet.
- 2026-09-21: metrics.jsonl opens in append mode.
  Relaunching a run under an existing name concatenates onto the old rows, so delete the run directory on the volume first.

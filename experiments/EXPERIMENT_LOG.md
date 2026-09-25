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
| screen_lr_1e-4 | 1e-4 | 3.002 | 2500 | 157 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_3e-4 | 3e-4 | 2.171 | 2500 | 156 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_1e-3 | 1e-3 | 1.641 | 2500 | 155 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_2e-3 | 2e-3 | 1.577 | 2500 | 159 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget, 50 eval batches |
| screen_lr_3e-3 | 3e-3 | 1.630 | 2500 | 158 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget, 20 eval batches, last eval at 2250 |
| screen_lr_5e-3 | 5e-3 | 1.867 | 2500 | 173 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget, 50 eval batches; stable but falling behind |
| screen_lr_1e-2 | 1e-2 | 3.676 | 2500 | 158 s | deleted; metrics.jsonl on cs336-runs volume | quarter budget; diverged after step 250, peak 4.44 at 1250 |
| lr_1e-3 | 1e-3 | 1.422 | 10000 | 598 s | [1r3q6wna](https://wandb.ai/porcini-labs/cs336-basics/runs/1r3q6wna) | full budget; under 1.45 from step 7000 |
| lr_2e-3 | 2e-3 | **1.389** | 10000 | 627 s | [l6c01mkt](https://wandb.ai/porcini-labs/cs336-basics/runs/l6c01mkt) | full budget; under 1.45 from step 6000; **base model** |
| lr_3e-3 | 3e-3 | 1.502 | 10000 | 635 s | [7bpj7o2f](https://wandb.ai/porcini-labs/cs336-basics/runs/7bpj7o2f) | full budget; never reaches 1.45 |
| lr_3e-4 | 3e-4 | 1.770 | 10000 | 643 s | [sk424k96](https://wandb.ai/porcini-labs/cs336-basics/runs/sk424k96) | full budget; too slow, flat from step 7000 |
| lr_5e-3 | 5e-3 | 2.132 | 10000 | 652 s | [flo7oqfs](https://wandb.ai/porcini-labs/cs336-basics/runs/flo7oqfs) | full budget; unstable: min 2.08 at step 1000, rises to 2.56 at 4000, partial recovery |
| lr_1e-2 | 1e-2 | 4.197 | 10000 | 653 s | [bzsuieni](https://wandb.ai/porcini-labs/cs336-basics/runs/bzsuieni) | full budget; diverged: min 2.51 at step 250, then collapses to 4.2 to 4.8 for the rest of training |
| lr_5e-2 | 5e-2 | 3.794 | 10000 | 648 s | [aronchts](https://wandb.ai/porcini-labs/cs336-basics/runs/aronchts) | full budget; diverged in warmup, loss 5.7 at step 500, never below 3.8 |

Search strategy: half-decade log grid over two decades around the Kingma et al. default of 1e-3 at a quarter budget, refine at a factor of 1.5 to 2 around the winner, then run the winner and its two neighbors at full budget.

Validation loss by step, all full-budget runs:

| step | 3e-4 | 1e-3 | 2e-3 | 3e-3 | 5e-3 | 1e-2 | 5e-2 |
|---|---|---|---|---|---|---|---|
| 250 | 3.415 | 2.718 | 2.493 | 2.418 | 2.397 | 2.513 | 4.864 |
| 1000 | 2.324 | 1.871 | 1.850 | 1.904 | 2.081 | 4.682 | 5.149 |
| 2000 | 2.044 | 1.702 | 1.694 | 1.782 | 2.229 | 4.636 | 4.832 |
| 4000 | 1.852 | 1.556 | 1.546 | 1.667 | 2.555 | 4.752 | 4.169 |
| 6000 | 1.782 | 1.463 | 1.446 | 1.567 | 2.309 | 4.352 | 3.888 |
| 8000 | 1.770 | 1.428 | 1.400 | 1.516 | 2.161 | 4.218 | 3.809 |
| 10000 | 1.770 | 1.422 | 1.389 | 1.502 | 2.132 | 4.197 | 3.794 |

Part (b), edge of stability: the best rate is 2e-3.
3e-3 is still stable but worse.
5e-3 is the first unstable rate: it is ahead of every other run at step 250, then loses ground from step 1000 to 4000 while the schedule is near its peak, and only recovers as the cosine decays.
1e-2 and 5e-2 diverge outright.
Divergence here does not mean NaN: gradient clipping at 1.0 caps every update, so the model collapses to a loss around 4 (roughly unigram level) and stays there rather than overflowing.
The edge of stability therefore sits between 3e-3 and 5e-3, and the best rate is about half of it.
Rates just below the edge win early (5e-3 and 3e-3 lead 2e-3 at step 250) but lose over the full schedule, because time spent near the peak rate is what destabilizes them, and the full-budget cosine holds the peak for far longer than the quarter-budget screen did.
That is also why 5e-3 looked fine at quarter budget (1.867) but not at full budget (2.132).

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

### lr_{3e-4,5e-3,1e-2,5e-2}  (2026-09-23)
Hypothesis:  The three finalists are too close to answer part (b); full-budget curves across two decades will show the slow side, the edge of stability, and at least one divergent run.
Command:     GPU=B200 modal run --detach modal_train.py --lrs 3e-4,5e-3,1e-2,5e-2 --extra "--train-iters 10000 --batch-size 128 --dtype bfloat16 --eval-interval 250 --eval-iters 50"
Result:      final val loss 3e-4 = 1.770, 5e-3 = 2.132, 1e-2 = 4.197, 5e-2 = 3.794; about 10.8 min each on B200; commit 802947b plus the uncommitted final-eval block.
Observation: 3e-4 is simply slow and flattens at 1.77 from step 7000.
             5e-3 is the first unstable rate: best of all runs at step 250, then degrades from step 1000 to 4000 and partially recovers as the rate decays.
             1e-2 collapses after step 250 and sits at 4.2 to 4.8 for the whole run; 5e-2 collapses during warmup.
             No run produced NaN because gradient clipping caps the update size, so divergence shows up as a collapse to unigram-level loss instead.
             The 5e-2 run drifts slowly downward late in training as the rate decays, so a collapsed model can still make progress once the rate is small enough.
Next:        Learning-rate problem is complete: seven full-budget curves, two divergent, best rate 2e-3 at about half the edge of stability.
             Batch size sweep next at lr 2e-3.

## 7.2.2 batch_size_experiment

Summary

| run | batch size | lr max | final val loss | steps | tokens seen | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|---|---|
| bs_1 | 1 | 2e-3 | 3.454 | 40000 | 10.2M (3%) | 15.4 min | [y1aqhvdl](https://wandb.ai/porcini-labs/cs336-basics/runs/y1aqhvdl) | step-capped; full budget would take 5.3 h |
| bs_8 | 8 | 2e-3 | 2.588 | 40000 | 81.9M (25%) | 14.9 min | [vdu8xlgn](https://wandb.ai/porcini-labs/cs336-basics/runs/vdu8xlgn) | step-capped; full budget would take 63 min |
| bs_32 | 32 | 2e-3 | 1.929 | 40000 | 327.7M | 16.2 min | [be6ewuv0](https://wandb.ai/porcini-labs/cs336-basics/runs/be6ewuv0) | full budget; unstable early (val rises from 10% to 25% of the run) |
| bs_64 | 64 | 2e-3 | 1.591 | 20000 | 327.7M | 12.0 min | [plcnpgsu](https://wandb.ai/porcini-labs/cs336-basics/runs/plcnpgsu) | full budget |
| lr_2e-3 | 128 | 2e-3 | 1.389 | 10000 | 327.7M | 10.4 min | [l6c01mkt](https://wandb.ai/porcini-labs/cs336-basics/runs/l6c01mkt) | full budget; reused from the lr sweep |
| bs_256 | 256 | 2e-3 | **1.383** | 5000 | 327.7M | 10.0 min | [46wfnyp0](https://wandb.ai/porcini-labs/cs336-basics/runs/46wfnyp0) | full budget; best, within noise of 128 |
| bs_512 | 512 | 2e-3 | 1.409 | 2500 | 327.7M | 9.6 min | [kh20fin9](https://wandb.ai/porcini-labs/cs336-basics/runs/kh20fin9) | full budget |
| bs_1024 | 1024 | 2e-3 | 1.479 | 1250 | 327.7M | 9.3 min | [045jo4dk](https://wandb.ai/porcini-labs/cs336-basics/runs/045jo4dk) | full budget |
| bs_2048 | 2048 | 2e-3 | 1.599 | 625 | 327.7M | 9.4 min | [sxnl32qq](https://wandb.ai/porcini-labs/cs336-basics/runs/sxnl32qq) | full budget; GPU memory limit (177 GB of 180) |
| bs32_lr_5e-4 | 32 | 5e-4 | 1.524 | 40000 | 327.7M | 13.3 min | [k1ob2y1j](https://wandb.ai/porcini-labs/cs336-basics/runs/k1ob2y1j) | lr re-check |
| bs32_lr_1e-3 | 32 | 1e-3 | 1.462 | 40000 | 327.7M | 16.3 min | [vlizbqnr](https://wandb.ai/porcini-labs/cs336-basics/runs/vlizbqnr) | lr re-check; best for batch 32 |
| bs2048_lr_5e-3 | 2048 | 5e-3 | 1.542 | 625 | 327.7M | 9.6 min | [mp97qa1y](https://wandb.ai/porcini-labs/cs336-basics/runs/mp97qa1y) | lr re-check; best for batch 2048 |
| bs2048_lr_1e-2 | 2048 | 1e-2 | 1.909 | 625 | 327.7M | 9.8 min | [eqyp59a0](https://wandb.ai/porcini-labs/cs336-basics/runs/eqyp59a0) | lr re-check; too high |

Protocol: fixed token budget of 327.68M for batch 32 and up, so steps scale inversely with batch size.
Batch 1 and 8 cannot reach the budget in reasonable time, so they ran for 40,000 steps each and are reported with the tokens they reached.
Every run evaluates on the same 50 batches of 128 sequences (`--eval-batch-size 128`), 40 evals per run, lr 2e-3 for all.

Throughput and memory (B200, bf16, from the 50-step probe and the full runs):

| batch | ms/step | tokens/s | peak GPU memory |
|---|---|---|---|
| 1 | 23 | 11k | 4 GB |
| 8 | 22 | 92k | 4 GB |
| 32 | 24 | 340k | 6 GB |
| 64 | 36 | 458k | 7 GB |
| 128 | 62 | 530k | 12 GB |
| 256 | 119 | 550k | 25 GB |
| 512 | 228 | 574k | 48 GB |
| 1024 | 443 | 592k | 95 GB |
| 2048 | 892 | 588k | 177 GB |
| 4096 | | | out of memory |

Findings:
- Below batch 32 the step time is flat at about 22 ms, so the GPU is launch-bound and tokens per second scale linearly with batch size.
  Batch 1 processes 11k tokens/s and would need 5.3 hours for the budget; batch 8 would need an hour.
  Throughput saturates between 128 and 256 at about 550k to 590k tokens/s, so beyond that a larger batch buys no wall-clock at fixed tokens.
- At fixed tokens and a fixed lr of 2e-3, final loss is U-shaped in batch size with the minimum at 128 to 256 (1.389 and 1.383, tied within noise).
  Larger batches are step-limited: batch 2048 gets only 625 updates, and its curve is still falling steeply at the end.
  Smaller batches are noisier at this lr: batch 32's val loss rises between 10% and 25% of the run while the schedule is near its peak, the same signature as an lr just past the edge of stability.
- Both sides of the U were partly an lr artifact, since 2e-3 was tuned at batch 128.
  Re-tuned, batch 32 improves from 1.929 to 1.462 at lr 1e-3, and batch 2048 from 1.599 to 1.542 at lr 5e-3.
  The best lr scales with batch size in the expected direction (1e-3 at 32, 2e-3 at 128, 5e-3 at 2048), roughly a factor of 2 to 2.5 per factor of 16 in batch, which is much slower than linear scaling.
- With each batch size at its best lr, the U is shallower but still there: 1.462 at 32, 1.389 at 128, 1.383 at 256, 1.542 at 2048.
  Batch 128 to 256 remains the sweet spot at this budget.
  The remaining penalty at batch 32 is real noise-limited progress per token; the penalty at 2048 is the step limit, since even a well-tuned run cannot converge in 625 updates.
- At a matched 74M tokens, batch 128 and 256 are also best (1.66), batch 64 is 1.85, batch 8 is 2.59, so larger batches are not more sample-efficient here either.
- Bigger is not always better: batch 2048 fits in memory and runs as fast as 256 per token, but at this budget it is 0.22 worse because it cannot take enough steps.

Entries

### probe_bs_{1..8192}  (2026-09-23)
Hypothesis:  Per-step time has a floor at small batch and throughput saturates somewhere below the memory limit; find both before spending on full runs.
Command:     GPU=B200 modal run modal_train.py --probe --sweep batch-size=1,8,32,64,256,512,1024,2048,4096,8192 --extra "--lrmax 2e-3 --dtype bfloat16"
Result:      table above; 4096 and 8192 out of memory at the first forward pass; about $0.50 total.
Observation: 2048 is the largest batch that fits (177 GB).
             Throughput plateaus at about 600k tokens/s from batch 128 up.
Next:        Fixed-token sweep for 32 through 2048; step-capped runs for 1 and 8.

### bs_{32,64,256,512,1024,2048} and bs_{1,8}  (2026-09-23)
Hypothesis:  At fixed tokens, larger batches lose sample efficiency once past the critical batch size; small batches are sample-efficient but too slow to be practical.
Command:     GPU=B200 modal run --detach modal_train.py --sweep batch-size=32,64,256,512,1024,2048 --fixed-tokens 327680000 --evals-per-run 40 --extra "--lrmax 2e-3 --dtype bfloat16 --eval-iters 50 --chkpt-interval 100000"
             GPU=B200 modal run --detach modal_train.py --sweep batch-size=1,8 --evals-per-run 40 --extra "--lrmax 2e-3 --dtype bfloat16 --eval-iters 50 --chkpt-interval 100000 --train-iters 40000"
Result:      summary table above; all eight runs finished; about $10 total.
Observation: The U-shape was not the expected monotone sample-efficiency curve: small batches did worse at equal tokens, not better.
             The batch 32 curve is non-monotone early, which points at lr 2e-3 being too high for its noisier gradients rather than at a sample-efficiency effect.
             Large batches lose because 625 to 1250 updates are not enough to converge under any lr, and the curves are still steep at the end.
Next:        Check whether the U is an lr artifact: batch 32 at 5e-4 and 1e-3, batch 2048 at 5e-3 and 1e-2 (four runs, about $4).
             Base config stays batch 128, lr 2e-3, since 256 is tied and 128 already has the finished checkpoint.

### bs32_lr_{5e-4,1e-3} and bs2048_lr_{5e-3,1e-2}  (2026-09-24)
Hypothesis:  The U in final loss versus batch size is partly because lr 2e-3 is too high for batch 32 and too low for batch 2048.
Command:     GPU=B200 modal run --detach modal_train.py --prefix bs32_ --sweep lrmax=5e-4,1e-3 --fixed-tokens 327680000 --evals-per-run 40 --extra "--batch-size 32 --dtype bfloat16 --eval-iters 50 --chkpt-interval 100000"
             GPU=B200 modal run --detach modal_train.py --prefix bs2048_ --sweep lrmax=5e-3,1e-2 --fixed-tokens 327680000 --evals-per-run 40 --extra "--batch-size 2048 --dtype bfloat16 --eval-iters 50 --chkpt-interval 100000"
Result:      batch 32: 5e-4 = 1.524, 1e-3 = 1.462 (was 1.929 at 2e-3); batch 2048: 5e-3 = 1.542, 1e-2 = 1.909 (was 1.599 at 2e-3); about $4.
Observation: The batch 32 curve at 1e-3 is monotone, so the early rise at 2e-3 was instability from the noisier gradients, not a property of small batches.
             Batch 2048 at 5e-3 is behind 2e-3 for the first half of the run and only pulls ahead as the rate decays, so a still-higher peak with a longer warmup might do slightly better, but 1e-2 already collapses early.
             Neither retuned extreme comes within 0.07 of batch 128 at 2e-3.
Next:        Batch size problem complete. Base config stays batch 128, lr 2e-3.
             Generation from the lr_2e-3 checkpoint next, then the ablations.

## 7.2.3 generate

Checkpoint used: lr_2e-3 at iteration 10000 (val loss 1.389), pulled from the cs336-runs volume.
Script: `uv run python cs336_basics/generate.py --run lr_2e-3 --pull --prompt "Once upon a time" --max-new-tokens 256`.
Decoding: prompt "Once upon a time", exactly 256 new tokens per sample with no early stop, so story boundaries appear inline as `<|endoftext|>`.
Model run on CPU in float32, seed 0 for every setting, about 4.5 s per sample.
Settings: greedy; ancestral sampling at temperature 1.0 and 0.7; nucleus sampling with top-p 0.9 at temperature 1.0 and 0.8.

Samples (2026-09-24)

#### greedy

```
Once upon a time, there was a little girl named Lily. She loved to play with her toys and have fun. One day, she found a big box in her room. She was very excited to see what was inside.
Lily opened the box and found a small, cute kitten. The kitten was very cute and soft. Lily loved the kitten and wanted to keep it as a pet. She took the kitten home and gave it a name. She named the kitten Fluffy.
Lily and Fluffy played together every day. They had so much fun with the kitten. They were very happy and became best friends. And they lived happily ever after.
<|endoftext|>
Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, Lily found a big, red apple in her kitchen. She was very happy and wanted to eat it.
Lily's mom saw her and said, "Lily, you must eat your food before you eat it." Lily did not want to eat her food, but she knew she had to listen to her mom. So, she ate her food very fast.
After eating, Lily's mom gave her a big hug. She said, "Lily, you are a good girl for eating
```

#### sample, temperature 1.0

```
Once upon a time, there was a little boy named Tim. Tim had a yummy cauliflower. It was his favorite vegetable. He ate every day to take care of it with care.
One day, Tim's mom made him a yummy cauliflower. But Tim did not want to eat it. He wanted to eat it now. His mom said, "Take care, Tim. Keep eating your cauliflower."
One sunny day, while Tim was eating the cauliflower, his friend Lily came over. She looked at the cauliflower and said, "Tim, you cannot eat vegetables like this. They are bad for you."
But Tim did not listen. He kept eating the cauliflower. After he ate it, something unexpected happened. Tim's cauliflower broke into pieces. Tim was very sad. He learned that he should listen to his friends who care about him.
<|endoftext|>
One day, a proud panda named Nana went to the park. Nana liked to watch the kids play and observe their toys. She was a big, happy panda.
Tom saw a little girl named Lily. Lily was sad and weeping. picks her up, a big boy named Tom, and started to dance. But Lily's heart was break. She ran away in tears.
Mana saw Lily and wanted to help.
```

#### sample, temperature 0.7

```
Once upon a time, there was a little boy named Tim. Tim had a big board. He liked to play with his board in his room. The board was in his room, and it was very comfortable.
One day, Tim wanted to play with his board. He put the board on the floor and pushed it. He made a fun sound with the board. Tim's mom saw him playing and said, "Tim, please be careful with the board."
Tim did not listen. He kept playing with his board. He hit the board and it broke. Tim was sad because he could not play with his board anymore. The board was gone, and Tim's day was not fun.
<|endoftext|>
One day, a little girl named Lily went to her garden. She saw a big tree with a hole in it. Lily wanted to fill the hole with water. She thought it would be fun to fill the hole with water.
Lily found a cup and filled it with water. She was very happy. But then, a fierce dog came and barked at her. Lily was scared and ran away.
Lily sneezed a time, but the dog did not go away. Instead, the dog left the water in the hole. Lily was sad, but she knew she had to
```

#### nucleus, top-p 0.9, temperature 1.0

```
Once upon a time, there was a little boy named Tim. Tim had a mysterious board. It was big and had many colors. Tim loved to draw on the board with his colorful chalk.
One day, Tim went to the park with his mom. He took his chalk with him. At the park, he saw a big slide. Tim wanted to slide down it. He ran to the top and sat down.
Tim was so happy to slide down the slide. He went up and down again. His mom watched him and smiled. They had a fun day at the park.
<|endoftext|>
Once upon a time, there was a boy named Tim. Tim had a toy cat. The cat was a big, red ball. Tim loved the cat very much. They played together every day.
One day, Tim and the cat went to the park. The sun was very hot. Tim put the cat on a cooler to keep it warm. The cat was grateful for the day. They played and played all day.
When it was time to go home, Tim said goodbye to the cat. The cat meowed and walked away. Tim was happy to have a new friend. He knew they would play together again soon.
<|endoftext|>
Once upon a time, in a small
```

#### nucleus, top-p 0.9, temperature 0.8

```
Once upon a time, there was a little boy named Tim. Tim had a big board. He liked to play with his board all day. One day, Tim was playing with his board in the park. He was having fun.
A girl named Sue came to the park. She saw Tim's board and wanted to play too. "Can I play with your board?" she asked. Tim said, "Yes, but be careful not to spoil it." Sue played with the board and had a lot of fun.
While they were playing, a dog came running up to them. The dog wanted to play too! The dog jumped on the board and made it go up and down. Tim and Sue were scared. But then, the dog started to dance! Tim and Sue were happy and laughed. The dog danced with them and they all had a great day.
<|endoftext|>
Once upon a time, in a small town, there lived a little girl named Mia. Mia loved to play outside and see the sky. One day, she saw a big cloud in the sky. She thought it was a tornado, but she was not scared.
Mia had an idea. She wanted to build a new house with the tornado. She asked her mom, "Can I build a new
```

Commentary:
- Greedy decoding is fluent but collapses onto the single most likely story.
  Both stories open with "a little girl named Lily" and the second is nearly a template of the first, and it produces the one outright contradiction ("you must eat your food before you eat it").
- Ancestral sampling at temperature 1.0 is the most diverse and the most error-prone.
  The first story is coherent; the second breaks down grammatically ("picks her up, a big boy named Tom", "Lily's heart was break") and invents names mid-story.
- Temperature 0.7 and nucleus sampling sit in between: varied across settings, grammatical throughout, with occasional semantic slips ("put the cat on a cooler to keep it warm", "build a new house with the tornado").
- Nucleus at top-p 0.9 and temperature 0.8 gave the best sample: two complete, internally consistent stories with dialogue and a resolution.
- All settings reproduce the TinyStories register (short sentences, a named child, a small problem, a moral) and stop cleanly at story ends, which is at or above the handout's reference sample.
- Note on the decode implementation: the greedy branch checks for temperature 0 after the logits are already divided by temperature, so temperature 0 yields infinities; use `method="greedy"` at any nonzero temperature instead.

## 7.3.1 layer_norm_ablation

Summary

| run | RMSNorm | lr max | final val loss | diverged? | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|---|
| lr_2e-3 | yes (pre) | 2e-3 | 1.389 | no | 10.4 min | [l6c01mkt](https://wandb.ai/porcini-labs/cs336-basics/runs/l6c01mkt) | base model |
| lr_1e-3 | yes (pre) | 1e-3 | 1.422 | no | 10.0 min | [1r3q6wna](https://wandb.ai/porcini-labs/cs336-basics/runs/1r3q6wna) | base architecture at the no-norm optimum, for a matched-lr comparison |
| nonorm_lr_2e-3 | none | 2e-3 | NaN | yes, step 390 | 9.3 min | [8451w95b](https://wandb.ai/porcini-labs/cs336-basics/runs/8451w95b) | previous optimal lr; loss overflows to NaN during warmup |
| nonorm_lr_1e-3 | none | 1e-3 | **1.428** | no (one spike at step 1000, recovered) | 9.1 min | [58i6o4mc](https://wandb.ai/porcini-labs/cs336-basics/runs/58i6o4mc) | best no-norm run |
| nonorm_lr_5e-4 | none | 5e-4 | 1.526 | no | 9.1 min | [tvsbnmzd](https://wandb.ai/porcini-labs/cs336-basics/runs/tvsbnmzd) | stable but too slow |

All runs: batch 128, 10,000 steps, bf16, eval on 50 batches of 128 every 250 steps, commit 2f67eac (`--norm none`).

Validation loss by step:

| step | base 2e-3 | base 1e-3 | no-norm 2e-3 | no-norm 1e-3 | no-norm 5e-4 |
|---|---|---|---|---|---|
| 250 | 2.493 | 2.718 | 2.572 | 2.789 | 3.135 |
| 500 | 2.077 | 2.190 | NaN | 2.259 | 2.486 |
| 1000 | 1.850 | 1.871 | NaN | 3.6e5 (spike) | 2.080 |
| 2000 | 1.694 | 1.702 | NaN | 1.786 | 1.852 |
| 4000 | 1.546 | 1.556 | NaN | 1.623 | 1.676 |
| 6000 | 1.446 | 1.463 | NaN | 1.505 | 1.570 |
| 8000 | 1.400 | 1.428 | NaN | 1.440 | 1.533 |
| 10000 | 1.389 | 1.422 | NaN | 1.428 | 1.526 |

Commentary:
- At the previous optimal lr of 2e-3 the model without RMSNorm diverges during warmup.
  The training loss is already 1.4e15 at step 380 and NaN from step 390, as the rate approaches its peak.
  This is a different failure from the high-lr runs with norm, which collapsed to a finite unigram-level loss because gradient clipping bounded every update.
  Without normalization the residual stream's scale is unbounded, so the forward pass itself overflows and clipping cannot save it.
- Halving the lr restores stability: at 1e-3 the no-norm model reaches 1.428, only 0.04 behind the base model at 2e-3 and within 0.006 of the base architecture at the same lr (1.422).
  It survived one transient blow-up at step 1000 (val 3.6e5) and recovered within a few hundred steps, so 1e-3 is close to its own edge of stability.
- At 5e-4 the no-norm model is stable throughout but 0.14 worse, so its usable lr range is narrow: a factor of 2 separates divergence from noticeably slow.
- RMSNorm's main contribution at this scale is therefore stability rather than raw quality.
  At a matched, stable lr it changes the final loss by less than eval noise; what it buys is the ability to train at a 2x higher rate without overflow and a wider margin around the optimum.

Entries

### nonorm_lr_{2e-3,1e-3,5e-4}  (2026-09-24)
Hypothesis:  Removing every RMSNorm will make the base lr of 2e-3 unstable; a lower lr will train but end worse.
Command:     GPU=B200 modal run --detach modal_train.py --prefix nonorm_ --sweep lrmax=2e-3,1e-3,5e-4 --extra "--norm none --batch-size 128 --train-iters 10000 --dtype bfloat16 --eval-interval 250 --eval-iters 50"
Result:      2e-3 NaN at step 390; 1e-3 = 1.428; 5e-4 = 1.526; about 9 min each, about $3; commit 2f67eac.
Observation: Divergence without norm is a forward-pass overflow, not the clipped collapse seen with norm.
             The lr 1e-3 run's spike at step 1000 and full recovery is worth a figure: it shows the no-norm model living right at its stability edge.
             The matched-lr gap (1.428 vs 1.422) is smaller than expected; the cost of removing norm is almost entirely the halved lr.
             No-norm runs are about 2% faster per step (53 vs 54 ms) since the norms are gone, not enough to matter.
Next:        Post-norm ablation with --norm post at lr 2e-3.

## 7.3.2 pre_norm_ablation

Summary

| run | norm placement | lr max | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|---|
| lr_2e-3 | pre | 2e-3 | **1.389** | 10.4 min | [l6c01mkt](https://wandb.ai/porcini-labs/cs336-basics/runs/l6c01mkt) | base model |
| postnorm_lr_2e-3 | post | 2e-3 | 1.405 | 10.7 min | [616njnll](https://wandb.ai/porcini-labs/cs336-basics/runs/616njnll) | stable; 0.016 behind pre-norm |
| lr_1e-3 | pre | 1e-3 | 1.422 | 10.0 min | [1r3q6wna](https://wandb.ai/porcini-labs/cs336-basics/runs/1r3q6wna) | from the lr sweep |
| postnorm_lr_1e-3 | post | 1e-3 | 1.412 | 10.1 min | [bzfj385m](https://wandb.ai/porcini-labs/cs336-basics/runs/bzfj385m) | stable; 0.010 ahead of pre-norm |

All runs: batch 128, 10,000 steps, bf16, eval on 50 batches of 128 every 250 steps, commit 2f67eac (`--norm post`).
Post-norm follows the handout's equations 27 and 28: RMSNorm after each residual add, final norm kept.

Validation loss by step:

| step | pre 2e-3 | post 2e-3 | pre 1e-3 | post 1e-3 |
|---|---|---|---|---|
| 250 | 2.493 | 2.513 | 2.718 | 2.706 |
| 1000 | 1.850 | 1.880 | 1.871 | 1.870 |
| 2000 | 1.694 | 1.728 | 1.702 | 1.710 |
| 4000 | 1.546 | 1.587 | 1.556 | 1.567 |
| 6000 | 1.446 | 1.480 | 1.463 | 1.465 |
| 8000 | 1.400 | 1.422 | 1.428 | 1.421 |
| 10000 | 1.389 | 1.405 | 1.422 | 1.412 |

Commentary:
- Post-norm trains stably at both rates, with no spikes: the largest post-warmup training loss is 2.42 at 2e-3, identical to pre-norm.
  The textbook post-norm instability did not appear at this depth.
- At the base lr of 2e-3, post-norm is 0.016 behind pre-norm and trails at every eval from step 250 on, so the gap is small but consistent.
  At 1e-3 the order flips and post-norm is 0.010 ahead, which is within eval noise.
  Overall, norm placement is worth at most a hundredth or two at 4 layers with warmup, far less than removing the norm entirely (see 7.3.1).
- The pre-norm advantage is expected to grow with depth: with post-norm the residual stream is renormalized at every block, so gradients to early layers pass through every norm, while pre-norm keeps a clean identity path.
  With only 4 blocks and 300 warmup steps the model is too shallow for that to bite.
- Pre-norm is the better default here on the strength of the 2e-3 comparison, but this ablation would need a deeper model or no warmup to reproduce the large gap the literature reports.

Entries

### postnorm_lr_{2e-3,1e-3}  (2026-09-25)
Hypothesis:  Post-norm will be unstable or clearly worse at 2e-3, and closer to pre-norm at a lower rate.
Command:     GPU=B200 modal run --detach modal_train.py --prefix postnorm_ --sweep lrmax=2e-3,1e-3 --extra "--norm post --batch-size 128 --train-iters 10000 --dtype bfloat16 --eval-interval 250 --eval-iters 50"
Result:      post-norm 2e-3 = 1.405 (pre 1.389); post-norm 1e-3 = 1.412 (pre 1.422); about 10.5 min each, about $2; commit 2f67eac.
Observation: No instability at either rate, and the gap to pre-norm is a hundredth or two in either direction.
             Post-norm ran about 10% slower per step (60 to 63 ms vs 54), which is more than the extra work explains; possibly GPU variance between hosts.
Next:        NoPE ablation: remove RoPE from attention and train at 2e-3.

## 7.3.3 no_pos_emb

Summary

| run | positional encoding | final val loss | wall-clock | wandb | notes |
|---|---|---|---|---|---|
| lr_2e-3 | RoPE | **1.389** | 10.4 min | [l6c01mkt](https://wandb.ai/porcini-labs/cs336-basics/runs/l6c01mkt) | base model |
| nope_lr_2e-3 | none (NoPE) | 1.439 | 9.7 min | [srrz9kw4](https://wandb.ai/porcini-labs/cs336-basics/runs/srrz9kw4) | stable; 0.051 behind at the end |

Both runs: batch 128, lr 2e-3, 10,000 steps, bf16, eval on 50 batches of 128 every 250 steps, commit 36f0fa5 (`--pos-emb none`).

Validation loss by step and the NoPE gap:

| step | RoPE | NoPE | gap |
|---|---|---|---|
| 250 | 2.493 | 2.875 | +0.382 |
| 500 | 2.077 | 2.258 | +0.181 |
| 1000 | 1.850 | 1.961 | +0.111 |
| 2000 | 1.694 | 1.758 | +0.065 |
| 4000 | 1.546 | 1.604 | +0.057 |
| 6000 | 1.446 | 1.500 | +0.053 |
| 8000 | 1.400 | 1.453 | +0.052 |
| 10000 | 1.389 | 1.439 | +0.051 |

Commentary:
- NoPE trains stably at the base lr and ends 0.051 behind RoPE, a clear and consistent gap, about five times the eval noise and comparable to the entire benefit of tuning the learning rate from 1e-3 to 2e-3.
- The gap is largest early (0.38 at step 250) and shrinks fast, settling near 0.05 from step 2000 on.
  Early in training the NoPE model has no direct way to tell positions apart and has to learn one; RoPE provides it for free.
  Once learned, the residual disadvantage is small but never closes.
- A causal decoder without position embeddings is not position-blind: the causal mask lets attention infer relative position from how many tokens are visible, and the model recovers most of the RoPE performance that way.
  That is why NoPE is far better than the 0.38 early gap would suggest, and why the mechanism is known to work at all.
- RoPE's remaining advantage is worth keeping: it costs nothing in parameters and about 5% in step time here (54 vs 57 ms is within host variance), for a 0.05 loss improvement at this budget.

Entries

### nope_lr_2e-3  (2026-09-25)
Hypothesis:  Without positional embeddings the model will still train, since the causal mask leaks position, but end noticeably worse than with RoPE.
Command:     GPU=B200 modal run --detach modal_train.py --run-name nope_lr_2e-3 --extra "--pos-emb none --lrmax 2e-3 --batch-size 128 --train-iters 10000 --dtype bfloat16 --eval-interval 250 --eval-iters 50"
Result:      NoPE 1.439 vs RoPE 1.389; about 10 min, about $1; commit 36f0fa5.
Observation: Gap of 0.05 at the end, consistent from step 2000 on; no instability (max post-warmup train loss 2.72 vs 2.42).
             The early gap (0.38 at step 250) is the model learning position from scratch.
Next:        SwiGLU vs SiLU ablation: implement the SiLU feed-forward variant with d_ff sized to match parameters, then train at 2e-3.

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

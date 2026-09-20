# Assignment 1 Experiment Log

All runs are tracked in wandb project `cs336-basics`.
Each run also writes `config.json` and `metrics.jsonl` under `checkpoints/<run_name>/`.

## Setup

- Hardware:
- dtype:
- Tokenizer / vocab:
- Base config (TinyStories): vocab 10000, ctx 256, 4 layers, d_model 512, 16 heads, d_ff 1344, RoPE theta 10000
- Eval protocol: eval every N steps on M batches of the validation set (fill in N, M and note the run-to-run noise you measured)

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
| | | | | | | |

Entries

### 

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

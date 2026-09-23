import argparse
import os
import numpy as np
import torch
import wandb
from typing import Tuple
from dataclasses import dataclass, fields, asdict
import time
import json

from transformer import *

class RunLogger:
    def __init__(self, run_dir: str, use_wandb: bool):
        self.use_wandb = use_wandb
        self.start = time.time()
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a")

    def log(self, step: int, **metrics):
        metrics["step"] = step
        metrics["wall_clock"] = time.time() - self.start
        self.f.write(json.dumps(metrics) + "\n")
        self.f.flush()
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def close(self):
        self.f.close()
        if self.use_wandb:
            wandb.finish()


@dataclass
class Config:
    train_path: str
    val_path: str

    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float

    Tw: float
    Tc: float
    lrmin: float
    lrmax: float

    betas: Tuple[float, float]
    weight_decay: float

    batch_size: int
    train_iters: int
    eval_interval: int
    eval_iters: int # batches to average val loss over
    eval_batch_size: int # sequences per eval batch, independent of the training batch size
    log_interval: int # how often to log train loss

    eps: float
    grad_clip_norm: float

    seed: int
    run_name: str
    device: str
    resume_path: str
    chkpt_dir: str
    chkpt_interval: int
    dtype: torch.dtype

    wandb_project: str
    no_wandb: bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    d = p.add_argument_group("data")
    d.add_argument("--train-path", type=str, required=True)
    d.add_argument("--val-path", type=str, required=True)

    m = p.add_argument_group("model")
    m.add_argument("--vocab-size", type=int, default=10000)
    m.add_argument("--context-length", type=int, default=256)
    m.add_argument("--num-layers", type=int, default=4)
    m.add_argument("--d-model", type=int, default=512)
    m.add_argument("--num-heads", type=int, default=16)
    m.add_argument("--d-ff", type=int, default=1344)
    m.add_argument("--rope-theta", type=float, default=10000.0)

    s = p.add_argument_group("schedule")
    s.add_argument("--lrmax", type=float, default=1e-3, help="peak learning rate (Kingma et al. default)")
    s.add_argument("--lrmin", type=float, default=None, help="final learning rate, defaults to lrmax / 10")
    s.add_argument("--Tw", type=float, default=None, help="warmup steps, defaults to 3%% of train-iters")
    s.add_argument("--Tc", type=float, default=None, help="cosine cycle length, defaults to train-iters")

    o = p.add_argument_group("optimizer")
    o.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--eps", type=float, default=1e-8)
    o.add_argument("--grad-clip-norm", type=float, default=1.0)

    t = p.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--train-iters", type=int, default=5000)
    t.add_argument("--eval-interval", type=int, default=200)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--eval-batch-size", type=int, default=128)
    t.add_argument("--log-interval", type=int, default=10)

    r = p.add_argument_group("runtime")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--run_name", type=str, default="run")
    r.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")
    r.add_argument("--resume-path", type=str, default=None)
    r.add_argument("--chkpt-dir", type=str, default="checkpoints")
    r.add_argument("--chkpt-interval", type=int, default=1000)
    r.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])

    w = p.add_argument_group('logging')
    w.add_argument("--wandb-project", type=str, default="cs336-basics")
    w.add_argument("--no-wandb", action="store_true")

    return p

def config_from_args(args: argparse.Namespace) -> Config:
    ns = vars(args)
    kwargs = {f.name: ns[f.name] for f in fields(Config)}
    # argparse nargs gives a list; Config declares Tuple[float, float]
    kwargs["betas"] = tuple(kwargs["betas"])
    # argparse gives a str; torch factory fns need a real torch.dtype
    kwargs["dtype"] = getattr(torch, kwargs["dtype"])
    if kwargs["Tc"] is None:
        kwargs["Tc"] = kwargs["train_iters"]
    if kwargs["Tw"] is None:
        kwargs["Tw"] = int(0.03 * kwargs["train_iters"])
    if kwargs["lrmin"] is None:
        kwargs["lrmin"] = kwargs["lrmax"] / 10

    return Config(**kwargs)


@torch.no_grad()
def estimate_loss(data, model, config):
    eval_iters = config.eval_iters
    split_loss = {}
    model.eval()
    for split in ['train', 'val']:
        dataset = data[split]
        losses = []
        for _ in range(eval_iters):
            X, y = get_batch(dataset, config.eval_batch_size, config.context_length, config.device)
            logits = model(X)
            loss = cross_entropy_with_logits(logits, y)
            losses.append(loss.item())
        split_loss[split] = np.mean(losses)
    model.train()
    return split_loss



if __name__ == "__main__": 
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Token budget: {config.batch_size * config.train_iters * config.context_length:,}")


    chkpt_run_dir = os.path.join(config.chkpt_dir, config.run_name)
    os.makedirs(chkpt_run_dir, exist_ok=True)
    wandb_config = asdict(config)
    # asdict carries a torch.dtype, which is not JSON serializable
    wandb_config["dtype"] = str(wandb_config["dtype"])

    if not config.no_wandb:
        wandb.init(project=config.wandb_project, name=config.run_name, config=wandb_config)
        wandb.define_metric("wall_clock")
        wandb.define_metric("eval/*", step_metric="wall_clock")

    with open(os.path.join(chkpt_run_dir, "config.json"), "w") as f:
        json.dump(wandb_config, f, indent=2)

    logger = RunLogger(chkpt_run_dir, use_wandb=not config.no_wandb)


    model = TransformerLM(
        vocab_size = config.vocab_size, 
        context_length = config.context_length,
        num_layers = config.num_layers, 
        d_model = config.d_model, 
        num_heads = config.num_heads, 
        d_ff = config.d_ff, 
        rope_theta = config.rope_theta, 
        device = config.device, 
        dtype = config.dtype
    )
    model.to(config.device)

    optimizer = AdamW(
        model.parameters(), 
        betas=config.betas,
        lr=config.lrmax, 
        weight_decay=config.weight_decay,
        eps=config.eps
    )

    if args.resume_path is not None:
        start = load_checkpoint(args.resume_path, model, optimizer)
    else:
        start = 0

    data = {}
    data['train'] = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    data['val'] = np.memmap(config.val_path, dtype=np.uint16, mode='r')

    for t in range(start, config.train_iters):
        if t % config.eval_interval == 0:
            print(f"Evaluating at iteration {t}")
            split_loss = estimate_loss(data, model, config)
            print(f"Train loss: {split_loss['train']} and val loss {split_loss['val']}")
            logger.log(t, **{"eval/train_loss": split_loss["train"], "eval/val_loss": split_loss["val"]})



        # This is an interesting way to assign the lr after already initializing adam. 
        lr = cosine_lr_scheduler(t, config.lrmax, config.lrmin, config.Tw, config.Tc)
        for group in optimizer.param_groups:
            group['lr'] = lr

        X, y = get_batch(data['train'], config.batch_size, config.context_length, config.device)
        logits = model(X)
        loss = cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        if t % config.log_interval == 0:
            print(f"Loss at iteration {t}: {loss:.3f}")
            tokens_seen = (t + 1) * config.batch_size * config.context_length
            logger.log(t, **{"train/loss": loss.item(), "train/lr": lr, "train/tokens": tokens_seen})

        loss.backward()
        clip_gradients(model.parameters(), config.grad_clip_norm, config.eps)

        optimizer.step()

        if t > 0 and t % config.chkpt_interval == 0: # Model checkpointing
            chkpt_filepath = os.path.join(chkpt_run_dir, f"iteration_{t}")
            print(f"Saving checkpoint at iter {t} to {chkpt_filepath}")
            save_checkpoint(model, optimizer, t, chkpt_filepath)

    split_loss = estimate_loss(data, model, config)
    print(f"Final train loss: {split_loss['train']} and val loss {split_loss['val']}")
    logger.log(config.train_iters, **{"eval/train_loss": split_loss["train"], "eval/val_loss":
    split_loss["val"]})



    chkpt_filepath = os.path.join(chkpt_run_dir, f"iteration_{config.train_iters}")
    print(f"Saving checkpoint at end of training {config.train_iters} to {chkpt_filepath}")
    save_checkpoint(model, optimizer, config.train_iters, chkpt_filepath)

    logger.close()




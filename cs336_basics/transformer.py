import torch
import torch.nn as nn
import numpy as np
import math
from einops import einsum, rearrange

def cross_entropy_with_logits(inputs, targets):
    max_elem = torch.amax(inputs, dim=-1)
    logsum = torch.log(torch.sum(torch.exp(inputs - max_elem.unsqueeze(-1)), dim=-1))
    target_logit = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
    losses = max_elem - target_logit + logsum
    return torch.mean(losses) #, dim=-1)


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype
        W = torch.empty((out_dim, in_dim), device=device, dtype=dtype)
        sdev = np.sqrt(2 / (in_dim + out_dim))
        clip = 3 * sdev
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(W, mean=0, std=sdev, a=-clip, b=clip))

    def forward(self, x):
        return einsum(self.weight, x, "out_dim in_dim, ... in_dim -> ... out_dim")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        W = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(W, mean=0, std=1, a=-3, b=3))

    def forward(self, x):
        return self.weight[x]

class RMSLayerNorm(nn.Module):
    def __init__(self, d_model, eps, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps).sqrt()
        result = x / rms * self.weight
        return result.to(in_dtype)

def silu(x):
    return torch.sigmoid(x) * x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None, gated=True):
        # gated=True: SwiGLU, w2(silu(w1 x) * w3 x), three weight matrices.
        # gated=False: plain SiLU FFN, w2(silu(w1 x)), two weight matrices (handout eq. 29); use d_ff = 4 * d_model to match params.
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gated = gated

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) if gated else None

    def forward(self, x):
        h = silu(self.w1(x))
        if self.gated:
            h = h * self.w3(x)
        return self.w2(h)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        assert d_k % 2 == 0, f"RoPE only operates on even dimensions"
        super().__init__()
        self.theta = theta
        self.d_k = d_k 
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = (1 / torch.pow(theta, 2 * torch.arange(d_k // 2, device=device).to(torch.float32) / d_k))
        angles = torch.outer(torch.arange(max_seq_len, device=device), inv_freq)

        self.register_buffer("Rcos", torch.cos(angles), persistent=False)
        self.register_buffer("Rsin", torch.sin(angles), persistent=False)


    def forward(self, x, token_positions):
        x_rearr = rearrange(x, "... seq_len (pair_d_k two) -> ... seq_len pair_d_k two", two=2)
        cos = self.Rcos[token_positions].to(x.dtype)
        sin = self.Rsin[token_positions].to(x.dtype)

        out_even = cos * x_rearr[..., 0] - sin * x_rearr[..., 1]
        out_odd = sin * x_rearr[..., 0] + cos* x_rearr[..., 1]
        out_stacked = torch.stack((out_even, out_odd), dim=-1)
        out = rearrange(out_stacked, "... seq_len half two -> ... seq_len (half two)")
        return out 


def softmax(in_features, dim):
    max_C = torch.amax(in_features, dim=dim, keepdim=True)
    exp = torch.exp(in_features - max_C)
    norm = exp / torch.sum(exp, dim=dim, keepdim=True)
    return norm


def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    inner = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    dp = inner / np.sqrt(d_k)
    dp = dp.masked_fill(~mask, float('-inf'))
    scores = softmax(dp, dim=-1)
    return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, device=None, dtype=None, rope=None):
        super().__init__()
        assert d_model % num_heads == 0, f"{d_model} needs to be divisible by {num_heads}."
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
#        self.register_buffer(
#            'mask',
#            torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=dtype, device=device)).to(bool)


    def forward(self, x):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

       # Reshape
        Q = rearrange(Q, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        seq_len = x.shape[-2]  

        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=x.device).expand(x.shape[:-1])
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask) 

        concat = rearrange(out, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        out = self.output_proj(concat)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None, rope=None, norm="pre", ffn="swiglu"):
        # norm: "pre" (RMSNorm before each sublayer), "post" (RMSNorm after each residual add), "none" (no RMSNorm)
        # ffn: "swiglu" (gated) or "silu" (ungated, handout eq. 29)
        super().__init__()
        if norm not in ("pre", "post", "none"):
            raise ValueError(f"norm must be pre, post or none, got {norm!r}")
        if ffn not in ("swiglu", "silu"):
            raise ValueError(f"ffn must be swiglu or silu, got {ffn!r}")
        self.norm = norm
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope, 
            device=device,
            dtype=dtype
        )

        self.ffn = FeedForwardNetwork(d_model, d_ff, device=device, dtype=dtype, gated=(ffn == "swiglu"))

        if norm == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSLayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)
            self.ln2 = RMSLayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x):
        if self.norm == "post":
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        else:  # pre-norm, or no norm at all (ln1/ln2 are identity)
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, device=None, dtype=None, norm="pre", pos_emb="rope", ffn="swiglu"):
        # pos_emb: "rope" (rotary embeddings on Q and K) or "none" (NoPE: no positional information beyond the causal mask)
        super().__init__()
        if pos_emb not in ("rope", "none"):
            raise ValueError(f"pos_emb must be rope or none, got {pos_emb!r}")
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length, device=device) if pos_emb == "rope" else None


        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, device, dtype, rope=self.rope, norm=norm, ffn=ffn) for _ in range(num_layers)])
        # "none" removes every RMSNorm in the model, including the final one; post-norm keeps the final norm as in pre-norm
        self.ln_final = nn.Identity() if norm == "none" else RMSLayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length


    def forward(self, x):
        embed = self.token_embeddings(x)
        layer = self.layers(embed)
        final_norm = self.ln_final(layer)
        logits = self.lm_head(final_norm)
        return logits



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), lr=1e-3, weight_decay=0.01, eps=1e-8):
        if lr < 0: 
            raise ValueError(f"Invalid learning rate {lr}")

        defaults = {
                'lr': lr, 
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay
                }

        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)

                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                
                m = state['m']
                
                if 'v' not in state:
                    state['v'] = torch.zeros_like(p)
                
                v = state['v']

                grad = p.grad.data
                alpha_t = lr * math.sqrt(1 - math.pow(beta2, t+1)) / (1 - math.pow(beta1, t+1))
                p.data -= weight_decay * lr * p.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2

                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss


def cosine_lr_scheduler(t, amax, amin, Tw, Tc):
    if t < Tw: 
        return (t / Tw) * amax
    if t > Tc: 
        return amin
    return amin + 0.5 * (amax - amin) * (1 + math.cos((t - Tw) * math.pi / (Tc - Tw)))

def clip_gradients(parameters, max_l2_norm, eps=1e-6):
    parameters = list(parameters)
    normsq = 0
    for p in parameters:
        if p.grad is None:
            continue
        normsq += torch.sum(torch.square(p.grad))

    norm = torch.sqrt(normsq)
    if norm <= max_l2_norm:
        return norm

    scale = max_l2_norm / (norm + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.mul_(scale)
    return norm


def save_checkpoint(model, optimizer, iteration, out):
    model_params = model.state_dict()
    optimizer_params = optimizer.state_dict()
    torch.save({
        'model': model_params,
        'optimizer': optimizer_params,
        'iteration': iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    ckpt = torch.load(src)
    model_params = ckpt['model']
    optimizer_params = ckpt['optimizer']
    iteration = ckpt['iteration']

    model.load_state_dict(model_params)
    optimizer.load_state_dict(optimizer_params)
    return iteration

def get_batch(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length)
    idx = starts[:, None] + offsets[None, :]
    X = torch.tensor(dataset[idx], device=device, dtype=torch.int64)
    y_idx = starts[:, None] + (offsets + 1)[None, :]
    y = torch.tensor(dataset[y_idx], device=device, dtype=torch.int64)
    return (X, y)

@torch.no_grad()
def decode(prompt, tokenizer, model, max_new_tokens, temperature=1.0, top_p=1.0, method='default'):
    encoded = tokenizer.encode(prompt)
    cl = model.context_length

    # special_tokens is a list of str; encode() takes one str at a time
    special_token_ids = set()
    if tokenizer.special_tokens is not None:
        special_token_ids = {tokenizer.encode(st)[0] for st in tokenizer.special_tokens}
    for i in range(max_new_tokens):
        batch_input = torch.unsqueeze(torch.tensor(encoded, dtype=torch.int64)[-cl:], 0)
        logits = model(batch_input)
        scaled_logits = logits[0, -1, :] / temperature
        probs = softmax(scaled_logits, dim=-1)
        if method == 'default': 
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        elif method == 'greedy' or temperature == 0: 
            next_token_id = torch.argmax(probs).item()
        elif method == 'nucleus' or method=='top_p': 
            ix = torch.argsort(probs)
            cum_prob = probs[ix].cumsum(dim=0)
            ixf = cum_prob > (1 - top_p)
            include_filter = torch.empty_like(ixf)
            include_filter[ix] = ixf
            filtered_probs = torch.where(include_filter, probs, 0)
            renormalized = filtered_probs / torch.sum(filtered_probs)
            next_token_id = torch.multinomial(renormalized, num_samples=1).item()

        encoded.append(next_token_id)
        if next_token_id in special_token_ids:
            break

    decoded = tokenizer.decode(encoded)
    print(decoded)
    return decoded


    
        








 



            
            








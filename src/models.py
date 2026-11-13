"""Independent categorical autoregressive implementation for the DSC5003 study.

Algorithm references (not a checkpoint-compatible port):
https://www.vldb.org/pvldb/vol13/p279-yang.pdf
https://github.com/naru-project/naru/blob/master/made.py
https://github.com/naru-project/naru/blob/master/estimators.py

Natural model-column order; binary input by default, with optional embeddings
for explicitly selected model-column indices. Categorical softmax output. NULL is a
regular modeled category but cannot satisfy ordinary comparisons. MASK (-1)
is a different, learned representation. Wildcard skipping is learned approximate
marginalization; disable it when testing sampling against the full joint model.
"""
import math
import operator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, input_degrees, output_degrees, *, strict=False, bias=True):
        super().__init__(len(input_degrees), len(output_degrees), bias=bias)
        src = torch.as_tensor(input_degrees)[None, :]
        dst = torch.as_tensor(output_degrees)[:, None]
        self.register_buffer('mask', (src < dst if strict else src <= dst).float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class ResidualBlock(nn.Module):
    def __init__(self, degrees):
        super().__init__()
        self.first = MaskedLinear(degrees, degrees)
        self.second = MaskedLinear(degrees, degrees)

    def forward(self, x):
        return x + self.second(F.relu(self.first(F.relu(x))))


class CategoricalMADE(nn.Module):
    def __init__(self, domain_sizes, hidden=256, layers=5, residual=False,
                 direct_io=False, column_masking=False, mask_seed=11123,
                 embedding_columns=None, embed_size=32):
        super().__init__()
        self.domain_sizes = tuple(int(n) for n in domain_sizes)
        if not self.domain_sizes or min(self.domain_sizes) < 1 or hidden < 1 or layers < 1:
            raise ValueError('positive domain sizes, width and depth required')
        self.ncols = len(self.domain_sizes)
        self.column_masking = column_masking
        self.residual = residual
        # Indices refer to the data/model order, after any external permutation.
        # No domain-size policy is applied here: the experiment config freezes it.
        selected = () if embedding_columns is None else tuple(embedding_columns)
        if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
               for i in selected):
            raise TypeError('embedding column indices must be integers')
        selected = tuple(int(i) for i in selected)
        if len(set(selected)) != len(selected) or any(i < 0 or i >= self.ncols for i in selected):
            raise ValueError('embedding column indices must be unique and in range')
        if not isinstance(embed_size, (int, np.integer)) or isinstance(embed_size, (bool, np.bool_)) or embed_size < 1:
            raise ValueError('positive integer embedding width required')
        self.embedding_columns = tuple(sorted(selected))
        self.embed_size = int(embed_size)
        self.input_widths = tuple(self.embed_size if i in self.embedding_columns
                                  else max(1, (n-1).bit_length())
                                  for i, n in enumerate(self.domain_sizes))
        inp = np.repeat(np.arange(self.ncols), self.input_widths)
        out = np.repeat(np.arange(self.ncols), self.domain_sizes)
        rng = np.random.RandomState(mask_seed)
        degrees = []
        for layer in range(layers):
            if self.ncols == 1:
                d = np.full(hidden, -1)
            elif residual:
                d = (np.arange(hidden)-1) % (self.ncols-1)
            else:
                lo = 0 if layer == 0 else int(degrees[-1].min())
                d = rng.randint(lo, self.ncols-1, size=hidden)
            degrees.append(d)
        self.input_layer = MaskedLinear(inp, degrees[0])
        self.hidden_layers = nn.ModuleList([
            ResidualBlock(degrees[i]) if residual else MaskedLinear(degrees[i-1], degrees[i])
            for i in range(1, layers)])
        self.output_layer = MaskedLinear(degrees[-1], out, strict=True)
        self.direct = MaskedLinear(inp, out, strict=True) if direct_io else None
        self.unknown = nn.ParameterList([nn.Parameter(torch.zeros(w)) for w in self.input_widths]) if column_masking else None
        for i, width in enumerate(self.input_widths):
            if i not in self.embedding_columns:
                self.register_buffer('bits_%d' % i, 1 << torch.arange(width))
        # Keep the historical binary construction and RNG stream untouched.
        # As with Linear, the trainer owns experiment-specific initialization;
        # its initialization pass sets embedding weights to normal(std=0.02).
        self.embeddings = nn.ModuleDict({str(i): nn.Embedding(self.domain_sizes[i], self.embed_size)
                                         for i in self.embedding_columns}) if self.embedding_columns else None

    def encode(self, x):
        """Encode validated codes; call validate_codes once per loaded batch/table.

        Keeping range checks separate avoids GPU synchronization on each sampling
        step. Only column-masking inference may use -1 (MASK), never as NULL.
        """
        if x.ndim != 2 or x.shape[1] != self.ncols:
            raise ValueError('x must have shape [batch, number of columns]')
        if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError('x must contain integer category codes')
        blocks = []
        for i, width in enumerate(self.input_widths):
            value = x[:, i:i+1].long()
            if i in self.embedding_columns:
                bits = self.embeddings[str(i)](value[:, 0].clamp_min(0))
            else:
                bits = ((value.clamp_min(0) & getattr(self, 'bits_%d' % i)) != 0).to(self.input_layer.weight.dtype)
            if self.column_masking:
                dropped = value < 0
                if self.training:
                    # Match the upstream distribution of per-column dropout
                    # probabilities; the random stream is PyTorch, not NumPy.
                    p = torch.randint(self.ncols, (), device=x.device).float() / self.ncols
                    dropped = dropped | (torch.rand(len(x), 1, device=x.device) < p)
                bits = torch.where(dropped, self.unknown[i][None, :], bits)
            blocks.append(bits)
        return torch.cat(blocks, dim=1)

    def validate_codes(self, x, *, allow_mask=False):
        if x.ndim != 2 or x.shape[1] != self.ncols:
            raise ValueError('invalid input shape')
        if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError('integer codes required')
        if allow_mask and not self.column_masking:
            raise ValueError('MASK requires column_masking')
        lower = -1 if allow_mask else 0
        upper = torch.tensor(self.domain_sizes, device=x.device)
        if bool(((x < lower) | (x >= upper)).any()):
            raise ValueError('category code out of range')

    def forward(self, x):
        encoded = self.encode(x)
        h = self.input_layer(encoded)
        if not self.residual:
            h = F.relu(h)
        for layer in self.hidden_layers:
            h = layer(h) if self.residual else F.relu(layer(h))
        logits = self.output_layer(h)
        if self.direct is not None:
            logits = logits + self.direct(encoded)
        return logits

    def split_logits(self, logits):
        return logits.split(self.domain_sizes, dim=-1)

    def nll(self, x):
        """Per-row negative log likelihood; train() enables column masking."""
        return sum(F.cross_entropy(logits, x[:, i].long(), reduction='none')
                   for i, logits in enumerate(self.split_logits(self(x))))


def _is_null(x):
    if x is None:
        return True
    if isinstance(x, np.datetime64):
        return bool(np.isnat(x))
    return isinstance(x, (float, np.floating)) and math.isnan(x)


def predicate_mask(domain, op=None, value=None, *, device=None):
    """Ordered original-value domain -> allowed categories; None op is wildcard.

    IS NULL/IS NOT NULL are stress-set extensions. Comparisons to NULL or with a
    NULL domain member never match. Domain ordering itself is not redefined here.
    """
    if op is None:
        valid = [True] * len(domain)
    elif op.upper() in ('IS NULL', 'IS NOT NULL'):
        valid = [_is_null(x) == (op.upper() == 'IS NULL') for x in domain]
    else:
        ops = {'=': operator.eq, '<': operator.lt, '<=': operator.le,
               '>': operator.gt, '>=': operator.ge}
        if op not in ops:
            raise ValueError('unsupported comparison: '+str(op))
        valid = [False if _is_null(x) or _is_null(value) else bool(ops[op](x, value)) for x in domain]
    return torch.tensor(valid, dtype=torch.bool, device=device)


@torch.no_grad()
def progressive_sample(model, allowed_masks, samples=2000, seed=0, wildcard_skipping=False):
    """Estimate selectivity, without cardinality multiplication or rounding.

    A None mask is wildcard. Each non-None mask is a bool vector in domain order.
    Zero probability paths get weight zero, with safe arbitrary continuation.
    No model parameters or global RNG state are changed. Restores training mode.
    """
    if len(allowed_masks) != model.ncols or samples < 1:
        raise ValueError('one mask per column and positive sample count required')
    if wildcard_skipping and not model.column_masking:
        raise ValueError('wildcard skipping requires a column-masking trained model')
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    masks = []
    for n, mask in zip(model.domain_sizes, allowed_masks):
        if mask is None:
            masks.append(None)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask.ndim != 1 or len(mask) != n:
                raise ValueError('predicate mask does not match column domain')
            if not bool(mask.any()):
                return 0.0
            masks.append(mask)
    if all(mask is None for mask in masks):
        return 1.0
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.zeros(samples, model.ncols, dtype=torch.long, device=device)
    if wildcard_skipping:
        for i, mask in enumerate(masks):
            if mask is None:
                x[:, i] = -1
    weight = torch.ones(samples, dtype=dtype, device=device)
    was_training = model.training
    model.eval()
    try:
        last = max(i for i, mask in enumerate(masks) if mask is not None)
        for i in range(last+1):
            if wildcard_skipping and masks[i] is None:
                continue
            probs = model.split_logits(model(x))[i].softmax(dim=1)
            restricted = probs if masks[i] is None else probs * masks[i][None, :]
            mass = restricted.sum(1)
            if masks[i] is not None:
                weight *= mass
            if i != last:
                safe = restricted.clone()
                dead = mass <= 0
                safe[dead, 0] = 1
                x[:, i] = torch.multinomial(safe, 1, generator=gen).squeeze(1)
        return weight.double().mean().item()
    finally:
        model.train(was_training)

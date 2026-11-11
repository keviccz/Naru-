"""Fixed-weight structural checks against vendored Naru; no fitting or updates.

Vendor reference: naru-project/naru @ 264cf4e9c96c9e34422f9eebc455a714aeef0b57.
The explicit test-only adapter keeps natural-order hidden masks and rebuilds only
direct I/O through the upstream general-order routine. This is necessary because
the natural-order direct mask uses a different encoded-column slice. No vendor
source is modified. This does NOT prove whole training trajectories identical.
"""
import argparse
import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import CategoricalMADE, MaskedLinear

VENDOR_COMMIT = '264cf4e9c96c9e34422f9eebc455a714aeef0b57'


def load_vendor():
    spec = importlib.util.spec_from_file_location('naru_reference_made', ROOT/'vendor/naru/made.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(device):
    reference = load_vendor()
    torch.set_num_threads(1)
    sizes = [2, 3, 5, 1]
    width, depth, mask_seed = 24, 5, 11123
    x = torch.tensor(list(itertools.product(*(range(n) for n in sizes))), dtype=torch.long, device=device)
    cases = []
    atol = 2e-5 if device == 'cuda' else 2e-6
    for residual, direct, masking in itertools.product((False, True), repeat=3):
        torch.manual_seed(42)
        np.random.seed(42)
        own = CategoricalMADE(sizes, hidden=width, layers=depth, residual=residual,
                              direct_io=direct, column_masking=masking, mask_seed=mask_seed).to(device).eval()
        # natural_ordering=False consumes a permutation before fixed_ordering is
        # applied. That changes MADE's random hidden degrees (not ResMADE's).
        # Initialize hidden masks through natural_ordering=True, then rebuild
        # direct I/O only through the generic routine. This is a test adapter.
        with contextlib.redirect_stdout(io.StringIO()):
            official = reference.MADE(len(sizes), [width]*depth, sum(sizes),
                                      natural_ordering=True, fixed_ordering=np.arange(len(sizes)),
                                      input_bins=sizes, input_encoding='binary', output_encoding='one_hot',
                                      residual_connections=residual, do_direct_io_connections=direct,
                                      column_masking=masking, seed=mask_seed)
        natural_mask = official.direct_io_layer.mask.clone() if direct else None
        official.natural_ordering = False
        if direct:
            official._build_or_update_direct_io()
        official = official.to(device).eval()
        ours_layers = [(name, layer) for name, layer in own.named_modules() if isinstance(layer, MaskedLinear)]
        reference_layers = [(name, layer) for name, layer in official.named_modules() if isinstance(layer, reference.MaskedLinear)]
        assert len(ours_layers) == len(reference_layers)
        layer_evidence = []
        with torch.no_grad():
            for (name, layer), (ref_name, ref_layer) in zip(ours_layers, reference_layers):
                assert layer.weight.shape == ref_layer.weight.shape, (name, ref_name)
                assert torch.equal(layer.mask, ref_layer.mask), (name, ref_name, 'mask mismatch')
                ref_layer.weight.copy_(layer.weight)
                assert (layer.bias is None) == (ref_layer.bias is None)
                if layer.bias is not None:
                    ref_layer.bias.copy_(layer.bias)
                layer_evidence.append({'self':name, 'official':ref_name, 'shape':list(layer.weight.shape),
                                       'mask_equal':True, 'active_edges':int(layer.mask.sum().item())})
            if masking:
                for left, right in zip(own.unknown, official.unk_embeddings):
                    right.copy_(left.reshape_as(right))
            own.validate_codes(x)
            encoded = own.encode(x)
            encoded_ref = official.EncodeInput(x)
            assert torch.equal(encoded, encoded_ref)
            logits = own(x)
            logits_ref = official(x)
            delta = (logits-logits_ref).abs().max().item()
            assert torch.allclose(logits, logits_ref, atol=atol, rtol=0), delta
            loss = own.nll(x)
            loss_ref = official.nll(logits_ref, x)
            nll_delta = (loss-loss_ref).abs().max().item()
            assert torch.allclose(loss, loss_ref, atol=atol, rtol=0), nll_delta
        training_checks = []
        for label, model in [('self', own), ('official', official)]:
            before = [p.detach().clone() for p in model.parameters()]
            torch.manual_seed(101)
            np.random.seed(101)
            model.train()
            model.zero_grad(set_to_none=True)
            loss_train = model.nll(x).mean() if label == 'self' else model.nll(model(x), x).mean()
            assert bool(torch.isfinite(loss_train))
            loss_train.backward()
            grads = [p.grad for p in model.parameters()]
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
            training_checks.append({'model':label, 'loss':float(loss_train.detach()),
                                    'gradient_tensors':len(grads), 'all_finite':True,
                                    'parameters_unchanged':True})
        discrepancy = None
        if direct:
            general = official.direct_io_layer.mask.detach().cpu()
            differing = torch.nonzero(natural_mask != general).tolist()
            discrepancy = {'natural_active_edges':int(natural_mask.sum()),
                           'general_active_edges':int(general.sum()),
                           'differing_edges':len(differing),
                           'first_differing_output_input_indices':differing[:12]}
            assert len(differing) > 0, 'expected encoded-domain natural direct-mask discrepancy not observed'
        cases.append({'residual':residual, 'direct_io':direct, 'column_masking':masking,
                      'linear_layers':layer_evidence, 'encoding_equal':True,
                      'logits_max_abs_delta':delta, 'nll_max_abs_delta':nll_delta,
                      'forward_backward':training_checks, 'natural_direct_mask_discrepancy':discrepancy})
    print(json.dumps({'status':'PASS', 'device':device, 'vendor_commit_expected':VENDOR_COMMIT,
                      'domain_sizes':sizes, 'hidden_width':width, 'hidden_layers_argument':depth,
                      'absolute_tolerance':atol, 'optimizer_steps':0,
                      'adapter':'natural_ordering=True initializes hidden masks; set natural_ordering=False and rebuild direct I/O only; fixed_ordering=np.arange(ncols) throughout',
                      'limitations':['Fixed weights compare structure, not training performance.',
                                     'Masking distributions match but NumPy/PyTorch random streams differ.',
                                     'Natural-order direct-mask discrepancy is reported; vendor source unchanged.'],
                      'cases':cases}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    main(parser.parse_args().device)

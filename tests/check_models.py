"""No training: enumerate tiny distributions and test the inference contract.

Run on the authorized server: python tests/check_models.py --device cpu
Optional --device cuda uses the externally selected CUDA_VISIBLE_DEVICES only.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.models import CategoricalMADE, predicate_mask, progressive_sample


def run(device):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    cases = []
    sizes = (2, 3, 2)
    table = torch.tensor(list(itertools.product(*map(range, sizes))), device=device)
    for residual in (False, True):
        for direct in (False, True):
            model = CategoricalMADE(sizes, hidden=24, layers=3, residual=residual,
                                    direct_io=direct, column_masking=True).to(device).double().eval()
            probs = model.nll(table).neg().exp()
            assert abs(probs.sum().item()-1) < 1e-10
            original = model.split_logits(model(table))
            # Changing current and future columns must leave each conditional unchanged.
            for i in range(len(sizes)):
                changed = table.clone()
                for j in range(i, len(sizes)):
                    changed[:, j] = (changed[:, j]+1) % sizes[j]
                actual = model.split_logits(model(changed))[i]
                assert torch.allclose(actual, original[i], atol=1e-12, rtol=0)
            queries = [
                [None, torch.tensor([False, True, True]), torch.tensor([False, True])],
                [torch.tensor([True, False]), None, torch.tensor([True, False])],
                [torch.tensor([True, False]), torch.tensor([False, True, False]), torch.tensor([False, True])],
            ]
            for masks in queries:
                keep = torch.ones(len(table), dtype=torch.bool, device=device)
                for i, mask in enumerate(masks):
                    if mask is not None:
                        keep &= mask.to(device)[table[:, i]]
                truth = probs[keep].sum().item()
                estimates = [progressive_sample(model, masks, samples=4096, seed=s) for s in range(6)]
                # Fixed absolute tolerance set before evaluating, for a probability
                # in [0,1]; catches incorrect weighting without demanding monotonic MC.
                error = abs(sum(estimates)/len(estimates)-truth)
                assert error < .012, (truth, estimates)
                cases.append({'residual':residual,'direct_io':direct,'exact':truth,'estimates':estimates,'mean_abs_error':error})
            assert progressive_sample(model, [None]*3) == 1
            assert progressive_sample(model, [torch.zeros(2, dtype=torch.bool), None, None]) == 0
            # Wildcard skipping on a factorized (all weights zero) model is exact.
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
            v = progressive_sample(model, [None, torch.tensor([False, True, False]), None], wildcard_skipping=True)
            assert abs(v-1/3) < 1e-12
    d = [None, 'A', 'B']
    assert predicate_mask(d, '<=', 'A').tolist() == [False, True, False]
    assert predicate_mask(d, '=', None).tolist() == [False]*3
    assert predicate_mask(d, 'IS NULL').tolist() == [True, False, False]
    assert predicate_mask(d).tolist() == [True]*3
    assert predicate_mask([None, float('nan'), 1.0], '>=', 0).tolist() == [False, False, True]
    one = CategoricalMADE([3], hidden=7, layers=2, residual=True).to(device).double().eval()
    x = torch.arange(3, device=device)[:, None]
    assert abs(one.nll(x).neg().exp().sum().item()-1) < 1e-12
    print(json.dumps({'status':'PASS','device':device,'training_performed':False,'enumeration_cases':cases}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    run(parser.parse_args().device)

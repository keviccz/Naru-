"""No fitting: legacy equivalence, mixed encoding and permutation correctness.

Run only on the authorized server. The baseline file is a frozen pre-change
source copy, not a second import of the modified implementation.
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from src.models import CategoricalMADE, progressive_sample


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('frozen_pre_improvement_models', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_states(sizes, device):
    return torch.tensor(list(itertools.product(*map(range, sizes))), device=device)


def permitted(table, masks):
    keep = torch.ones(len(table), dtype=torch.bool, device=table.device)
    for i, mask in enumerate(masks):
        if mask is not None:
            keep &= torch.as_tensor(mask, device=table.device)[table[:, i]]
    return keep


def mc_check(model, masks, truth, *, skipping, samples=4096, repeats=6, tolerance=.012):
    values = [progressive_sample(model, masks, samples=samples, seed=s,
                                 wildcard_skipping=skipping) for s in range(repeats)]
    error = abs(statistics.mean(values)-truth)
    assert error < tolerance, (truth, values, skipping, error, tolerance)
    return dict(exact=truth, mean=statistics.mean(values), mean_abs_error=error,
                tolerance=tolerance, skipping=skipping, samples=samples,
                repeats=repeats, estimates=values)


def check_legacy(old, device):
    results = []
    for residual, direct, masking in [(False, False, False), (False, True, True),
                                      (True, False, True), (True, True, True)]:
        for selected in [None, []]:
            kwargs = dict(hidden=17, layers=3, residual=residual, direct_io=direct,
                          column_masking=masking, mask_seed=41)
            torch.manual_seed(1234)
            reference = old.CategoricalMADE((2, 5, 3), **kwargs)
            reference_rng = torch.get_rng_state().clone()
            torch.manual_seed(1234)
            candidate = CategoricalMADE((2, 5, 3), embedding_columns=selected, **kwargs)
            assert torch.equal(reference_rng, torch.get_rng_state())
            before, after = reference.state_dict(), candidate.state_dict()
            assert list(before) == list(after)
            assert all(torch.equal(before[k], after[k]) for k in before)
            candidate.load_state_dict(before, strict=True)
            reference = reference.to(device).double()
            candidate = candidate.to(device).double()
            x = all_states((2, 5, 3), device)
            for train in [False, True]:
                reference.train(train)
                candidate.train(train)
                torch.manual_seed(5678)
                a = reference(x)
                cpu_rng, cuda_rng = torch.get_rng_state().clone(), None
                if device.type == 'cuda':
                    cuda_rng = torch.cuda.get_rng_state(device).clone()
                torch.manual_seed(5678)
                b = candidate(x)
                assert torch.equal(a, b)
                assert torch.equal(cpu_rng, torch.get_rng_state())
                if cuda_rng is not None:
                    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
                reference.zero_grad(set_to_none=True)
                candidate.zero_grad(set_to_none=True)
                a.square().sum().backward()
                b.square().sum().backward()
                for (ka, pa), (kb, pb) in zip(reference.named_parameters(), candidate.named_parameters()):
                    assert ka == kb and ((pa.grad is None and pb.grad is None) or
                                        (pa.grad is not None and pb.grad is not None and
                                         torch.equal(pa.grad, pb.grad)))
            results.append(dict(residual=residual, direct_io=direct, column_masking=masking,
                                embedding_columns=selected, constructor_rng_equal=True,
                                state_dict_equal=True, eval_train_logits_and_gradients_equal=True))
    return results


def check_variants(device):
    results = []
    original_sizes = (2, 65, 3)
    original_masks = [
        [None, [False]+[True]*64, [False, True, False]],
        [[True, False], None, [True, False, True]],
        [[False, True], [False, True]+[False]*62+[True], None],
    ]
    specs = [('V-order', (2, 0, 1), []), ('V-embed', (0, 1, 2), [1]),
             ('V-all', (2, 1, 0), [1])]
    for number, (name, order, embedding_columns) in enumerate(specs):
        sizes = tuple(original_sizes[i] for i in order)
        torch.manual_seed(200+number)
        model = CategoricalMADE(sizes, hidden=24, layers=3, residual=True,
                                direct_io=True, column_masking=True,
                                embedding_columns=embedding_columns, embed_size=32).to(device).double().eval()
        x = all_states(sizes, device)
        model.validate_codes(x)
        probabilities = model.nll(x).neg().exp()
        assert abs(probabilities.sum().item()-1) < 1e-10
        original_logits = model.split_logits(model(x))
        for i in range(3):
            changed = x.clone()
            for j in range(i, 3):
                changed[:, j] = (changed[:, j]+1) % sizes[j]
            actual = model.split_logits(model(changed))[i]
            assert torch.equal(actual, original_logits[i])
            for j in embedding_columns:
                gradient = torch.autograd.grad(original_logits[i].sum(),
                                               model.embeddings[str(j)].weight,
                                               retain_graph=True, allow_unused=True)[0]
                if j >= i:
                    assert gradient is None or torch.count_nonzero(gradient).item() == 0

        # Check MASK encoding itself: all-wildcard inference would bypass it.
        with torch.no_grad():
            for i, unknown in enumerate(model.unknown):
                unknown.copy_(-(i+1)-torch.arange(len(unknown), device=device, dtype=torch.double)/10)
            for j in embedding_columns:
                model.embeddings[str(j)].weight[0].copy_(
                    1+torch.arange(32, device=device, dtype=torch.double)/10)
        masked = torch.full((2, 3), -1, dtype=torch.long, device=device)
        model.validate_codes(masked, allow_mask=True)
        encoded_mask = model.encode(masked)
        expected_mask = torch.cat([u for u in model.unknown]).expand(2, -1)
        assert torch.equal(encoded_mask, expected_mask)
        zero_encoded = model.encode(torch.zeros_like(masked))
        offset = 0
        for i, width in enumerate(model.input_widths):
            if i in embedding_columns:
                assert width == 32
                assert torch.equal(zero_encoded[:, offset:offset+width],
                                   model.embeddings[str(i)].weight[0].expand(2, -1))
                assert not torch.equal(zero_encoded[:, offset:offset+width],
                                       encoded_mask[:, offset:offset+width])
            offset += width
        assert torch.equal(model.input_layer.mask, model.input_layer.mask.to(torch.bool).to(torch.double))

        # Unknown vectors do not enter ordinary, non-skipping joint probabilities.
        probabilities = model.nll(x).neg().exp()
        assert abs(probabilities.sum().item()-1) < 1e-10
        samples = []
        approximate_skipping = []
        for masks in original_masks:
            model_masks = [masks[i] for i in order]
            original_table = x[:, [order.index(i) for i in range(3)]]
            keep_original = permitted(original_table, masks)
            keep_model = permitted(x, model_masks)
            assert torch.equal(keep_original, keep_model)
            truth = probabilities[keep_model].sum().item()
            samples.append(mc_check(model, model_masks, truth, skipping=False))
            estimates = [progressive_sample(model, model_masks, samples=4096,
                                            seed=s, wildcard_skipping=True)
                         for s in range(3)]
            approximate_skipping.append(dict(
                model_joint_reference=truth, estimates=estimates,
                mean=statistics.mean(estimates),
                absolute_difference_from_joint=abs(statistics.mean(estimates)-truth),
                equality_required=False, samples=4096, repeats=3))
        assert progressive_sample(model, [None]*3, wildcard_skipping=True) == 1
        assert progressive_sample(model, [torch.zeros(sizes[0], dtype=torch.bool), None, None],
                                  wildcard_skipping=True) == 0
        # A factorized fixture makes skipping exact, independent of learned MASK.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        for i, size in enumerate(sizes):
            masks = [None]*3
            masks[i] = torch.arange(size) % 3 == 0
            truth = float(masks[i].sum())/size
            for skip in [False, True]:
                estimate = progressive_sample(model, masks, seed=0, wildcard_skipping=skip)
                assert abs(estimate-truth) < 1e-12
        results.append(dict(variant=name, column_order=list(order), model_domain_sizes=list(sizes),
                            embedding_columns=embedding_columns, normalized_joint=True,
                            current_future_perturbations_independent=True,
                            current_future_embedding_gradients_zero=True,
                            mask_encoding_checked=True, null_code0_distinct_from_mask=True,
                            factorized_single_column_skipping_exact=True,
                            correlated_model_no_skipping=samples,
                            untrained_correlated_skipping_diagnostics=approximate_skipping))
    return results


class ExactTableModel(nn.Module):
    """Exact conditional table, with explicit oracle marginalization of MASKs.

    This is a fixture, not the claim that a learned embedding implements an oracle.
    """
    def __init__(self, table, sizes):
        super().__init__()
        self.ncols = len(sizes)
        self.domain_sizes = tuple(sizes)
        self.column_masking = True
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.double, device=table.device))
        self.register_buffer('table', table)

    def split_logits(self, x):
        return x.split(self.domain_sizes, dim=-1)

    def forward(self, x):
        blocks = []
        valid = torch.ones((len(x), len(self.table)), dtype=torch.bool, device=x.device)
        for i, size in enumerate(self.domain_sizes):
            encoded = torch.nn.functional.one_hot(self.table[:, i], num_classes=size).double()
            count = valid.double() @ encoded
            total = count.sum(dim=1, keepdim=True)
            probabilities = torch.where(total > 0, count/total.clamp_min(1),
                                         torch.full_like(count, 1/size))
            blocks.append(probabilities.log())
            valid &= (x[:, i:i+1] < 0) | (x[:, i:i+1] == self.table[None, :, i])
        return torch.cat(blocks, dim=1)


def check_permutation_oracle(device):
    rows = [(0, 0, 0)]*5+[(0, 1, 1)]*3+[(1, 1, 0)]*2+[(1, 2, 1)]*6
    table = torch.tensor(rows, dtype=torch.long, device=device)
    masks_list = [
        [[True, False], [False, True, True], None],
        [None, [False, True, True], [False, True]],
        [[True, False], None, [False, True]],
        [[True, False], [False, False, True], None],
    ]
    results = []
    for order in itertools.permutations(range(3)):
        model = ExactTableModel(table[:, order], tuple((2, 3, 2)[i] for i in order))
        for qi, masks in enumerate(masks_list):
            truth = permitted(table, masks).double().mean().item()
            ordered = [masks[i] for i in order]
            assert torch.equal(permitted(table, masks), permitted(model.table, ordered))
            for skip in [False, True]:
                result = mc_check(model, ordered, truth, skipping=skip)
                results.append(dict(order=list(order), query=qi, oracle_fixture=True, **result))
    return results


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.device_count() != 1:
        raise RuntimeError('Select only the authorized physical GPU0 before this test')
    torch.set_num_threads(1)
    baseline = Path(args.baseline).resolve()
    assert baseline.is_file()
    old = load_baseline(baseline)
    result = dict(status='PASS', device=str(device), training_performed=False,
                  baseline_source=str(baseline),
                  default_legacy=check_legacy(old, device),
                  variants=check_variants(device),
                  exact_permutation_oracle=check_permutation_oracle(device),
                  skipping_scope='Exact checks use factorized/explicit oracle fixtures only; '
                                 'no equality claim for learned correlated wildcard marginalization')
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--baseline', default=str(ROOT/'experiments/improvement_20261111/code_reference/models_baseline.py'))
    parser.add_argument('--output')
    run(parser.parse_args())

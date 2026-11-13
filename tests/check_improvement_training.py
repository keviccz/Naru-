"""Remote GPU0 preflight: frozen epoch loop versus the new exact-step loop.

This is a correctness check, not a formal benchmark fit or a quality result.
Run with the frozen original 1M encoded table and an isolated experiment root.
No torch import or training occurs until main explicitly selects physical GPU0.
"""
import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    experiment = Path(args.experiment_root).resolve()
    output = Path(args.output).resolve() if args.output else experiment/'evidence/training_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Preserve prior receipt; choose a new --output: '+str(output))
    started = time.monotonic()
    receipt = {'status':'RUNNING', 'purpose':'training-loop correctness; not a benchmark fit',
               'experiment_root':str(experiment), 'encoded_dir':str(Path(args.encoded_dir).resolve()),
               'checks':[], 'physical_gpu_index':0}
    def save():
        tmp = output.with_suffix(output.suffix+'.tmp')
        tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(tmp, output)
    def passed(name, **evidence):
        receipt['checks'].append({'name':name, 'status':'PASS', **evidence})
        save()
        print(json.dumps(receipt['checks'][-1]), flush=True)
    save()
    try:
        if sys.platform != 'linux':
            raise RuntimeError('Training checks run only on the authorized Linux server')
        query = subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid,name','--format=csv,noheader'],
                               capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        fields = [x.strip() for x in query.split(',')]
        if len(fields) != 2 or 'RTX 3090' not in fields[1]:
            raise RuntimeError('Physical GPU0 is not the approved RTX3090: '+query)
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[0], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8')
        import numpy as np
        import torch
        from torch import nn
        sys.path.insert(0, str(ROOT))
        trainer = load_module('improvement_training_under_test', ROOT/'scripts/train_improvement.py')
        frozen_models = load_module('frozen_training_reference_models', experiment/'code_reference/models_baseline.py')
        frozen_trainer = load_module('frozen_training_reference_loop', experiment/'code_reference/train_baseline.py')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One visible CUDA GPU required; CPU fallback is forbidden')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        receipt.update(gpu_uuid=fields[0], gpu_name=fields[1], torch_version=torch.__version__,
                       numpy_version=np.__version__, deterministic_algorithms=True, tf32=False)

        def require_equal(a, b, path='root'):
            if torch.is_tensor(a):
                if not torch.is_tensor(b) or not torch.equal(a, b):
                    raise AssertionError('tensor mismatch at '+path)
            elif isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    raise AssertionError('array mismatch at '+path)
            elif isinstance(a, dict):
                if a.keys() != b.keys():
                    raise AssertionError('dictionary keys differ at '+path)
                for key in a:
                    require_equal(a[key], b[key], path+'.'+str(key))
            elif isinstance(a, (tuple, list)):
                if len(a) != len(b):
                    raise AssertionError('sequence length differs at '+path)
                for i, (x, y) in enumerate(zip(a, b)):
                    require_equal(x, y, path+'.'+str(i))
            elif a != b:
                raise AssertionError('value mismatch at '+path+': '+repr(a)+' != '+repr(b))

        def cpu_copy(value):
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k:cpu_copy(v) for k,v in value.items()}
            if isinstance(value, list):
                return [cpu_copy(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(v) for v in value)
            return copy.deepcopy(value)

        def rng_state():
            return {'python':random.getstate(), 'numpy':np.random.get_state(),
                    'torch_cpu':torch.get_rng_state().clone(),
                    'torch_cuda':[x.clone().cpu() for x in torch.cuda.get_rng_state_all()]}

        def snapshot(model, optimizer):
            return {'model':cpu_copy(model.state_dict()), 'optimizer':cpu_copy(optimizer.state_dict()),
                    'rng':rng_state()}

        def legacy_init(sizes, width, layers, seed):
            # Deliberately independent of the new initialization helper.
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = frozen_models.CategoricalMADE(sizes, hidden=width, layers=layers,
                         residual=True, direct_io=True, column_masking=True, mask_seed=seed)
            torch.manual_seed(seed)
            def initialize(module):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            model.apply(initialize)
            return model.to(device)

        def optimizer_for(model):
            return torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9,0.999), eps=1e-8)

        def run_legacy(table, sizes, width, layers, steps, batch, warmups, points):
            model = legacy_init(sizes, width, layers, 0)
            optimizer = optimizer_for(model)
            snapshots = {0:snapshot(model, optimizer)}
            losses = []; sizes_seen = []; learning_rates = []; rows_seen = 0
            row_rng = np.random.default_rng(1000003)
            step = 0
            # The original epoch loop, with a stop inserted immediately after
            # the requested update. No new permutation is consumed at the stop.
            while step < steps:
                torch.cuda.synchronize(device)
                ids = row_rng.permutation(len(table))
                model.train()
                for begin in range(0, len(table), batch):
                    raw = np.asarray(table[ids[begin:begin+batch]], dtype=np.int64)
                    xb = torch.from_numpy(raw).to(device)
                    step += 1
                    lr = frozen_trainer.learning_rate(step, warmups)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.nll(xb).mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('nonfinite legacy loss')
                    loss.backward(); optimizer.step()
                    losses.append(float(loss.detach())); sizes_seen.append(len(xb)); learning_rates.append(lr)
                    rows_seen += len(xb)
                    if step in points:
                        snapshots[step] = snapshot(model, optimizer)
                    if step == steps:
                        break
                torch.cuda.synchronize(device)
            return {'snapshots':snapshots, 'losses':losses, 'batch_rows':sizes_seen,
                    'lrs':learning_rates, 'rows_seen':rows_seen}

        def run_new(table, sizes, width, layers, steps, batch, warmups, points, window_steps):
            model = trainer.initialize_model(sizes, [], 0, width, layers, device)
            optimizer = optimizer_for(model)
            snapshots = {0:snapshot(model, optimizer)}
            losses = []; sizes_seen = []; learning_rates = []; windows = []
            def on_step(record):
                losses.append(record['loss']); sizes_seen.append(record['batch_rows']); learning_rates.append(record['lr'])
                if record['step'] in points:
                    snapshots[record['step']] = snapshot(model, optimizer)
            def on_window(record):
                windows.append(record.copy())
                before = rng_state()
                # Exercise the production checkpoint's CPU-state serialization.
                stream = io.BytesIO()
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'global_step':record['step']}, stream)
                require_equal(before, rng_state(), 'logging_rng')
            final = trainer.optimize(model, optimizer, table, list(range(len(sizes))), device,
                                     0, batch, steps, warmups, window_steps, on_step, on_window)
            return {'snapshots':snapshots, 'losses':losses, 'batch_rows':sizes_seen,
                    'lrs':learning_rates, 'rows_seen':final['rows_seen'], 'windows':windows, 'final':final}

        # Pure iterator proves short tails, real-epoch rollover, exact final
        # truncation, and no next-epoch permutation consumption after the stop.
        for steps in (1,3,6,8):
            actual_rng = np.random.default_rng(19)
            actual = list(trainer.iter_training_batches(10,4,steps,actual_rng))
            expected_rng = np.random.default_rng(19)
            expected = []
            while len(expected) < steps:
                ids = expected_rng.permutation(10)
                for begin in range(0,10,4):
                    expected.append(ids[begin:begin+4])
                    if len(expected) == steps:
                        break
            for row, ids in zip(actual, expected):
                require_equal(row['indices'], ids, 'iterator_indices')
            require_equal(actual_rng.bit_generator.state, expected_rng.bit_generator.state, 'iterator_rng')
            assert actual[-1]['step'] == steps and actual[-1]['final_step']
            assert [r['real_epoch'] for r in actual] == [1+i//3 for i in range(steps)]
        passed('tiny_iterator_exact_truncation', row_count=10, batch=4, budgets=[1,3,6,8],
               eight_step_batch_rows=[4,4,2,4,4,2,4,4], no_extra_permutation=True)
        for warmups in (0,8,8000):
            for step in (1,7,8,9,100,489,490,7999,8000,8001,122480):
                assert trainer.learning_rate(step,warmups) == frozen_trainer.learning_rate(step,warmups)
        passed('learning_rate_exact', warmups=[0,8,8000], includes_warmup_boundary=True)

        tiny_rng = np.random.default_rng(23)
        tiny = np.stack([tiny_rng.integers(0,d,10) for d in [3,5,2]],axis=1).astype(np.int64)
        old = run_legacy(tiny,[3,5,2],24,3,8,4,8,{1,3,6,8})
        new = run_new(tiny,[3,5,2],24,3,8,4,8,{1,3,6,8},3)
        for key in ('snapshots','losses','batch_rows','lrs','rows_seen'):
            require_equal(old[key],new[key],'tiny.'+key)
        assert new['batch_rows'] == [4,4,2,4,4,2,4,4]
        assert new['final']['step'] == 8 and new['final']['real_epoch'] == 3
        assert new['final']['real_epochs_completed'] == 2 and new['final']['rows_in_epoch'] == 8
        assert [w['step'] for w in new['windows']] == [3,6,8]
        begin = 0
        for window in new['windows']:
            end = window['step']
            expected = sum(x*n for x,n in zip(new['losses'][begin:end],new['batch_rows'][begin:end]))/sum(new['batch_rows'][begin:end])
            assert window['mean_training_nll_nats'] == expected
            begin = end
        passed('tiny_multiepoch_optimizer_exact', steps=8, rows_seen=28, final_real_epoch=3,
               completed_real_epochs=2, window_steps=[3,6,8], state_and_adam_and_rng_exact=True,
               row_weighted_logging_exact=True, checkpoint_logging_rng_unchanged=True)
        del old,new
        torch.cuda.empty_cache()

        encoded = Path(args.encoded_dir).resolve()
        table = np.load(encoded/'1000000.npy', mmap_mode='r', allow_pickle=False)
        domains = json.loads((encoded/'1000000_domains.json').read_text(encoding='utf-8'))
        sizes = [len(d) for d in domains['domains']]
        if table.shape != (1000000,len(sizes)) or table.dtype.kind not in 'iu':
            raise ValueError('The original complete encoded 1M table is required')
        points = {100,488,489,490}
        old = run_legacy(table,sizes,256,5,490,2048,8000,points)
        new = run_new(table,sizes,256,5,490,2048,8000,points,100)
        for key in ('snapshots','losses','batch_rows','lrs','rows_seen'):
            require_equal(old[key],new[key],'1M.'+key)
        assert len(new['losses']) == 490 and new['batch_rows'][488] == 576 and new['batch_rows'][489] == 2048
        assert new['final']['real_epoch'] == 2 and new['final']['rows_seen'] == 1002048
        passed('original_1M_seed0_first100_and_epoch_boundary_exact', updates=490,
               snapshot_steps=[0,100,488,489,490], maximum_absolute_loss_difference=0.0,
               all_model_buffers_parameters_adam_and_rng_exact=True,
               epoch1_steps=489, epoch1_tail_rows=576, step490_rows=2048,
               total_rows_seen=1002048, checkpoint_logging_rng_unchanged=True,
               source_models=str(experiment/'code_reference/models_baseline.py'),
               source_learning_rate=str(experiment/'code_reference/train_baseline.py'))
        receipt.update(status='PASS', all_checks_passed=True, elapsed_seconds=time.monotonic()-started,
                       formal_training_started=False)
        save()
        print(json.dumps({'status':'PASS','receipt':str(output),'elapsed_seconds':receipt['elapsed_seconds']}), flush=True)
        return 0
    except BaseException as error:
        receipt.update(status='FAILED', all_checks_passed=False, error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        save()
        print(receipt['traceback'], file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',required=True)
    parser.add_argument('--encoded-dir',required=True)
    parser.add_argument('--output')
    parser.add_argument('--threads',type=int,default=4)
    sys.exit(main(parser.parse_args()))

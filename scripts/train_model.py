"""Explicit remote-only Naru training; never starts at import time.

Uses the optimizer and LR formula from vendored naru/train_model.py at commit
264cf4e9c96c9e34422f9eebc455a714aeef0b57. Data loading, seed handling,
progress receipts and serialization are independent implementations.

Example (run-dir must not already exist):
python scripts/train_model.py --encoded-dir data/snapshot/encoded \
    --scale 1000000 --model own_resmade --seed 0 --run-dir experiments/run001
"""
import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
VENDOR_COMMIT = '264cf4e9c96c9e34422f9eebc455a714aeef0b57'


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def learning_rate(step, warmups):
    # Upstream MADE uses embed_size=32 even with binary inputs. Do NOT replace
    # this by hidden width. Its warmups==0 branch uses 1e-2, not base_lr=8e-4.
    return 32**-0.5 * min(step**-0.5, step * warmups**-1.5) if warmups else 1e-2


def main(args):
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status':'STARTING', 'started_at':utcnow(), 'pid':os.getpid(),
              'model':args.model, 'scale':args.scale, 'seed':args.seed,
              'epochs_requested':args.epochs, 'epochs_completed':0, 'steps_completed':0,
              'run_dir':str(out), 'checkpoint_policy':'rolling checkpoint.pt after each completed epoch'}
    write_json(out/'status.json', status)
    write_json(out/'arguments.json', vars(args))
    try:
        if sys.platform != 'linux':
            raise RuntimeError('This script is restricted to the authorized Linux training server')
        if min(args.epochs, args.batch, args.width, args.layers, args.log_every, args.threads) <= 0 or args.warmups < 0 or args.seed < 0:
            raise ValueError('positive dimensions/counts and nonnegative seed/warmups required')
        gpu_query = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        gpu_fields = [v.strip() for v in gpu_query.split(',')]
        if len(gpu_fields) != 5 or 'RTX 3090' not in gpu_fields[1]:
            raise RuntimeError('Physical GPU0 is not the approved RTX 3090: '+gpu_query)
        gpu_uuid = gpu_fields[0]
        previous_visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # No torch import occurs before the physical GPU0 UUID is selected.
        import numpy as np
        import torch
        from torch import nn
        sys.path.insert(0, str(ROOT))
        from src.models import CategoricalMADE
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU required; no CPU fallback')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        encoded = Path(args.encoded_dir).resolve()
        if args.purpose == 'benchmark':
            snapshot = json.loads((encoded.parent/'manifest.json').read_text())
            encoding = json.loads((encoded/'encoding_receipt.json').read_text())
            if snapshot['status'] != 'COMPLETE' or encoding['status'] != 'COMPLETE':
                raise RuntimeError('Complete frozen snapshot and encoding required')
        data_path = encoded/(args.scale+'.npy')
        domain_path = encoded/(args.scale+'_domains.json')
        metadata = json.loads(domain_path.read_text(encoding='utf-8'))
        sizes = [len(d) for d in metadata['domains']]
        columns = metadata['columns']
        table = np.load(data_path, mmap_mode='r', allow_pickle=False)
        if table.ndim != 2 or table.shape[1] != len(sizes) or len(columns) != len(sizes):
            raise ValueError('encoded table and domains have inconsistent dimensions')
        if table.dtype.kind not in 'iu' or table.shape[0] < 1 or min(sizes) < 1 or len(set(columns)) != len(columns):
            raise ValueError('invalid encoded data contract')
        n = len(table)
        if args.scale != 'full' and n != int(args.scale):
            raise ValueError('row count differs from requested scale')
        upper = np.asarray(sizes, dtype=np.int64)
        # Bounded CPU validation before GPU training; only one chunk is copied.
        for begin in range(0, n, 200000):
            chunk = table[begin:begin+200000]
            if np.any(chunk < 0) or np.any(chunk >= upper):
                raise ValueError('out-of-domain code at chunk starting row '+str(begin))
        residual = args.model != 'own_made'
        if args.model == 'official_resmade':
            spec = importlib.util.spec_from_file_location('naru_reference_made', ROOT/'vendor/naru/made.py')
            vendor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vendor)
            model = vendor.MADE(len(sizes), [args.width]*args.layers, sum(sizes),
                                input_bins=sizes, input_encoding='binary', output_encoding='one_hot',
                                embed_size=32, seed=args.seed, residual_connections=True,
                                do_direct_io_connections=True, column_masking=True,
                                natural_ordering=False, fixed_ordering=np.arange(len(sizes)))
        else:
            model = CategoricalMADE(sizes, hidden=args.width, layers=args.layers,
                                    residual=residual, direct_io=residual, column_masking=residual,
                                    mask_seed=args.seed)
        # Reset after constructors so final Linear initialization does not depend
        # on extra temporary modules allocated by the vendor constructor.
        torch.manual_seed(args.seed)
        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        model.apply(initialize)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
        batches = math.ceil(n/args.batch)
        config = {**vars(args), 'encoded_dir':str(encoded), 'data_path':str(data_path),
                  'domain_path':str(domain_path), 'rows':n, 'columns':columns, 'domain_sizes':sizes,
                  'data_dtype':str(table.dtype), 'data_bytes':data_path.stat().st_size,
                  'data_mtime_ns':data_path.stat().st_mtime_ns, 'batches_per_epoch':batches,
                  'column_order':list(range(len(sizes))), 'input_encoding':'binary', 'output_encoding':'one_hot',
                  'residual':residual, 'direct_io':residual, 'column_masking':residual,
                  'parameter_count':sum(p.numel() for p in model.parameters()),
                  'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                  'physical_gpu_index':0, 'gpu_uuid':gpu_uuid, 'gpu_preflight_csv':gpu_query,
                  'previous_cuda_visible_devices':previous_visibility,
                  'effective_cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                  'torch_version':torch.__version__, 'cuda_version':torch.version.cuda,
                  'numpy_version':np.__version__, 'python':sys.version, 'python_executable':sys.executable,
                  'optimizer':{'name':'Adam','betas':[0.9,0.999],'eps':1e-8,'weight_decay':0},
                  'lr_schedule':'32**(-0.5)*min(global_step**(-0.5),global_step*warmups**(-1.5)); warmups=0 => 0.01',
                  'initialization':'Xavier uniform Linear weights; zero biases; seed reset after construction',
                  'shuffle':'NumPy default_rng(seed+1000003), full row-id permutation per epoch; no full feature-tensor copy',
                  'deterministic_algorithms':True, 'tf32':False, 'precision':'float32',
                  'official_commit':VENDOR_COMMIT, 'checkpoint_compatible_with_upstream_cli':False,
                  'deviations':['Independent memmap trainer and seeded NumPy row shuffling.',
                                'All methods use the requested training seed, rather than upstream TrainTask hardcoded zero.',
                                'Natural column order is fixed for every seed.',
                                'Official ResMADE uses general-order direct-I/O masks without source edits.',
                                'Column-masking probability random streams differ between implementations.',
                                'Epoch losses are row-weighted, not equally weighted batch means.',
                                'Full-table final NLL is streamed without column masking or parameter updates.']}
        write_json(out/'config.json', config)
        source_dir = out/'sources'; source_dir.mkdir()
        shutil.copy2(__file__, source_dir/'train_model.py')
        shutil.copy2(ROOT/'src/models.py', source_dir/'own_models.py')
        if args.model == 'official_resmade':
            shutil.copy2(ROOT/'vendor/naru/made.py', source_dir/'official_made.py')
        shutil.copy2(domain_path, out/'domains.json')
        print(json.dumps({'event':'configured','config':config}), flush=True)
        row_rng = np.random.default_rng(args.seed+1000003)
        torch.cuda.reset_peak_memory_stats(device)
        status.update(status='RUNNING', phase='training', updated_at=utcnow())
        write_json(out/'status.json', status)
        global_step = 0
        def nll(x):
            return model.nll(model(x), x) if args.model == 'official_resmade' else model.nll(x)
        def batch_tensor(indices):
            # Fancy indexing materializes one CPU batch, then int64 transfer only.
            raw = np.asarray(table[indices], dtype=np.int64)
            return torch.from_numpy(raw).to(device)
        for epoch in range(args.epochs):
            torch.cuda.synchronize(device)
            epoch_start = time.monotonic()
            ids = row_rng.permutation(n)
            model.train()
            loss_sum = 0.0
            seen = 0
            for begin in range(0, n, args.batch):
                xb = batch_tensor(ids[begin:begin+args.batch])
                global_step += 1
                lr = learning_rate(global_step, args.warmups)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = nll(xb).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite loss at global step '+str(global_step))
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(xb)
                seen += len(xb)
                if global_step % args.log_every == 0:
                    status.update(steps_completed=global_step, epoch_in_progress=epoch+1,
                                  rows_in_epoch=seen, lr=lr, updated_at=utcnow(),
                                  elapsed_seconds=time.monotonic()-started)
                    write_json(out/'status.json', status)
                    print(json.dumps({'event':'progress',**status}), flush=True)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic()-epoch_start
            epoch_record = {'epoch':epoch+1,'rows':seen,'mean_training_nll_nats':loss_sum/seen,
                            'elapsed_seconds':elapsed,'rows_per_second':seen/elapsed,
                            'global_step':global_step,'last_lr':lr,
                            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),
                            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(device),
                            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            with (out/'epoch_metrics.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(epoch_record,allow_nan=False)+'\n')
            # Rolling checkpoint is atomic; old successful checkpoint survives a
            # failure while serializing the new one. No optimizer resume is offered.
            tmp_checkpoint = out/'checkpoint.pt.tmp'
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'config':config, 'epochs_completed':epoch+1, 'global_step':global_step,
                        'optimizer_state_saved':False}, tmp_checkpoint)
            os.replace(tmp_checkpoint,out/'checkpoint.pt')
            status.update(epochs_completed=epoch+1,steps_completed=global_step,
                          updated_at=utcnow(),elapsed_seconds=time.monotonic()-started,
                          checkpoint='checkpoint.pt',last_epoch=epoch_record)
            write_json(out/'status.json',status)
            print(json.dumps({'event':'epoch_complete',**epoch_record}),flush=True)
        status.update(phase='final_nll',updated_at=utcnow())
        write_json(out/'status.json',status)
        model.eval()
        total_nll = 0.0
        evaluation_start = time.monotonic()
        with torch.no_grad():
            for begin in range(0,n,args.batch):
                values = nll(batch_tensor(slice(begin,begin+args.batch)))
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError('nonfinite final evaluation NLL')
                total_nll += values.double().sum().item()
        torch.cuda.synchronize(device)
        result = {'final_full_table_nll_nats':total_nll/n,'evaluation_seconds':time.monotonic()-evaluation_start,
                  'training_epochs':args.epochs,'training_steps':global_step,'rows':n,
                  'checkpoint_bytes':(out/'checkpoint.pt').stat().st_size,
                  'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),
                  'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(device),
                  'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        write_json(out/'training_result.json',result)
        status.update(status='COMPLETE',phase='done',ended_at=utcnow(),exit_code=0,
                      elapsed_seconds=time.monotonic()-started,result=result)
        write_json(out/'status.json',status)
        print(json.dumps({'event':'complete',**status}),flush=True)
        return 0
    except BaseException as error:
        status.update(status='FAILED',ended_at=utcnow(),exit_code=1,
                      elapsed_seconds=time.monotonic()-started,
                      error_type=type(error).__name__,error=str(error))
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(out/'status.json',status)
        print(json.dumps(status),file=sys.stderr,flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--purpose', choices=['benchmark','synthetic_check'], default='benchmark')
    parser.add_argument('--encoded-dir',required=True)
    parser.add_argument('--scale',choices=['1000000','5000000','full'],default='1000000')
    parser.add_argument('--model',choices=['own_made','own_resmade','official_resmade'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch',type=int,default=2048)
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--layers',type=int,default=5)
    parser.add_argument('--warmups',type=int,default=8000)
    parser.add_argument('--log-every',type=int,default=100)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--run-dir',required=True)
    sys.exit(main(parser.parse_args()))

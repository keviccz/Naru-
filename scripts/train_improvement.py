"""Remote GPU0-only trainer for the four predeclared ResMADE ablations.

No model import or execution occurs at import time. The pure batch iterator and
optimizer loop are shared with the preflight equivalence checks.
"""
import argparse
import datetime as dt
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
TARGET_STEPS = 122480
VIRTUAL_WINDOW_STEPS = 6124
VARIANTS = ('order', 'steps', 'embed', 'all')


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def learning_rate(step, warmups):
    return 32**-0.5 * min(step**-0.5, step * warmups**-1.5) if warmups else 1e-2


def iter_training_batches(n, batch, max_steps, row_rng):
    """Yield the original epoch shuffle/short-tail sequence, truncated by step.

    No new permutation is consumed after the last requested update. Indices are
    views into this epoch's row permutation, just as in the baseline trainer.
    """
    if min(n, batch, max_steps) <= 0:
        raise ValueError('positive row, batch and step counts required')
    step = 0
    epoch = 0
    batches = math.ceil(n/batch)
    while step < max_steps:
        epoch += 1
        ids = row_rng.permutation(n)
        for batch_index, begin in enumerate(range(0, n, batch), 1):
            step += 1
            yield {'step': step, 'real_epoch': epoch,
                   'batch_in_epoch': batch_index, 'batches_per_epoch': batches,
                   'indices': ids[begin:begin+batch],
                   'rows_in_epoch': min(begin+batch, n),
                   'epoch_complete': begin+batch >= n,
                   'final_step': step == max_steps}
            if step == max_steps:
                break


def initialize_model(domain_sizes, embedding_columns, seed, width, layers, device):
    """Preserve baseline seed reset and Linear initialization order."""
    import numpy as np
    import torch
    from torch import nn
    from src.models import CategoricalMADE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = CategoricalMADE(domain_sizes, hidden=width, layers=layers,
                            residual=True, direct_io=True, column_masking=True,
                            mask_seed=seed, embedding_columns=embedding_columns,
                            embed_size=32)
    torch.manual_seed(seed)
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    model.apply(initialize)
    return model.to(device)


def optimize(model, optimizer, table, column_order, device, seed, batch,
             max_steps, warmups, window_steps=None, on_step=None, on_window=None):
    """Exactly one train-mode forward per update; callbacks receive scalars.

    window_steps=None emits real-epoch windows. An integer emits step windows.
    Logging cannot change the model RNG because it uses the loss already seen.
    """
    import numpy as np
    import torch
    n = len(table)
    natural = list(column_order) == list(range(table.shape[1]))
    row_rng = np.random.default_rng(seed+1000003)
    window_start = time.monotonic()
    loss_sum = 0.0
    window_rows = 0
    rows_seen = 0
    window = 0
    last = None
    for info in iter_training_batches(n, batch, max_steps, row_rng):
        if info['batch_in_epoch'] == 1:
            model.train()
        raw = np.asarray(table[info['indices']], dtype=np.int64)
        if not natural:
            raw = np.ascontiguousarray(raw[:, column_order])
        xb = torch.from_numpy(raw).to(device)
        step = info['step']
        lr = learning_rate(step, warmups)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = model.nll(xb).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite loss at global step '+str(step))
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        loss_sum += value * len(xb)
        window_rows += len(xb)
        rows_seen += len(xb)
        record = {k: v for k, v in info.items() if k != 'indices'}
        record.update(loss=value, lr=lr, batch_rows=len(xb), rows_seen=rows_seen,
                      real_epoch_progress=(info['real_epoch']-1)+info['rows_in_epoch']/n)
        if on_step is not None:
            on_step(record)
        boundary = (step % window_steps == 0) if window_steps else info['epoch_complete']
        if boundary or info['final_step']:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-window_start
            window += 1
            last = dict(record, epoch=window, virtual_epoch=window,
                        rows=window_rows, mean_training_nll_nats=loss_sum/window_rows,
                        elapsed_seconds=elapsed, rows_per_second=window_rows/elapsed,
                        global_step=step, last_lr=lr,
                        real_epochs_completed=info['real_epoch'] if info['epoch_complete'] else info['real_epoch']-1,
                        window_kind='optimizer_steps' if window_steps else 'real_epoch')
            if on_window is not None:
                on_window(last)
            window_start = time.monotonic()
            loss_sum = 0.0
            window_rows = 0
    return last


def resolve_order(args, ncols):
    natural = list(range(ncols))
    if args.variant not in ('order', 'all'):
        return natural, natural, None
    if not args.order_file:
        raise ValueError('order/all requires --order-file')
    source = Path(args.order_file).resolve()
    metadata = json.loads(source.read_text(encoding='utf-8'))
    order = metadata['order']
    if sorted(order) != natural or len(order) != ncols:
        raise ValueError('order must be a permutation of original columns')
    inverse = [order.index(i) for i in natural]
    if 'inverse_order' in metadata and metadata['inverse_order'] != inverse:
        raise ValueError('inverse_order disagrees with order')
    if 'scale' in metadata and str(metadata['scale']) != args.scale:
        raise ValueError('MI order belongs to a different scale')
    return order, inverse, source


def main(args):
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status':'STARTING', 'started_at':utcnow(), 'pid':os.getpid(),
              'model':'own_resmade', 'variant':args.variant, 'scale':args.scale,
              'seed':args.seed, 'run_dir':str(out), 'steps_completed':0,
              'epochs_completed':0, 'checkpoint_policy':'rolling final-state checkpoint at each reporting window; no optimizer resume'}
    write_json(out/'status.json', status)
    write_json(out/'arguments.json', vars(args))
    try:
        if sys.platform != 'linux':
            raise RuntimeError('Training is restricted to the authorized Linux server')
        if min(args.batch, args.width, args.layers, args.log_every, args.threads) <= 0 or args.warmups < 0 or args.seed < 0:
            raise ValueError('invalid dimensions/counts/seed/warmups')
        if args.purpose == 'benchmark' and ((args.batch,args.width,args.layers,args.warmups) != (2048,256,5,8000) or args.seed not in (0,1,2)):
            raise ValueError('formal ResMADE configuration and seeds are frozen; use synthetic_check for fixtures')
        query = subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True).stdout.strip()
        fields = [v.strip() for v in query.split(',')]
        if len(fields) != 5 or 'RTX 3090' not in fields[1]:
            raise RuntimeError('Physical GPU0 is not the approved RTX3090: '+query)
        previous_visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[0], CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        import numpy as np
        import torch
        sys.path.insert(0,str(ROOT))
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU is required; no CPU fallback')
        device = torch.device('cuda:0');torch.cuda.set_device(device)
        encoded = Path(args.encoded_dir).resolve()
        if args.purpose == 'benchmark':
            if json.loads((encoded.parent/'manifest.json').read_text())['status'] != 'COMPLETE' or json.loads((encoded/'encoding_receipt.json').read_text())['status'] != 'COMPLETE':
                raise RuntimeError('Frozen complete snapshot/encoding required')
        data_path = encoded/(args.scale+'.npy');domain_path=encoded/(args.scale+'_domains.json')
        metadata=json.loads(domain_path.read_text(encoding='utf-8'));original_sizes=[len(d) for d in metadata['domains']];original_columns=metadata['columns']
        table=np.load(data_path,mmap_mode='r',allow_pickle=False);n=len(table)
        if table.ndim != 2 or table.shape[1] != len(original_sizes) or len(set(original_columns)) != len(original_sizes) or table.dtype.kind not in 'iu' or n<1:
            raise ValueError('encoded table/domain contract mismatch')
        if args.scale != 'full' and n != int(args.scale):
            raise ValueError('row count differs from requested scale')
        upper=np.asarray(original_sizes,dtype=np.int64)
        for begin in range(0,n,200000):
            chunk=table[begin:begin+200000]
            if np.any(chunk<0) or np.any(chunk>=upper):raise ValueError('out-of-domain code')
        order,inverse,order_source=resolve_order(args,len(original_sizes));sizes=[original_sizes[i] for i in order];columns=[original_columns[i] for i in order]
        embedded=[k for k,i in enumerate(order) if original_sizes[i]>64] if args.variant in ('embed','all') else []
        batches=math.ceil(n/args.batch);step_variant=args.variant in ('steps','all')
        total_steps=TARGET_STEPS if step_variant else 20*batches
        if args.max_steps is not None:
            if args.purpose=='benchmark' and args.max_steps!=total_steps:raise ValueError('formal step budget is frozen')
            total_steps=args.max_steps
        if total_steps<=0:raise ValueError('positive total steps required')
        window_steps=VIRTUAL_WINDOW_STEPS if step_variant else None
        model=initialize_model(sizes,embedded,args.seed,args.width,args.layers,device)
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.9,0.999),eps=1e-8)
        config={**vars(args),'model':'own_resmade','encoded_dir':str(encoded),'run_dir':str(out),'data_path':str(data_path),'domain_path':str(domain_path),
                'rows':n,'original_columns':original_columns,'original_domain_sizes':original_sizes,'columns':columns,'domain_sizes':sizes,
                'column_order':order,'inverse_order':inverse,'order_file':str(order_source) if order_source else None,
                'embedding_columns':embedded,'embedding_original_columns':[order[k] for k in embedded],'embedding_column_names':[columns[k] for k in embedded],
                'embedding_threshold':64,'embed_size':32,'input_encoding':'mixed_binary_embedding' if embedded else 'binary','output_encoding':'one_hot',
                'residual':True,'direct_io':True,'column_masking':True,'parameter_count':sum(p.numel() for p in model.parameters()),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                'total_steps':total_steps,'training_steps_requested':total_steps,'epochs':20 if not step_variant else None,'batches_per_epoch':batches,
                'reporting_window_steps':window_steps,'reporting_points_expected':math.ceil(total_steps/window_steps) if window_steps else math.ceil(total_steps/batches),
                'data_dtype':str(table.dtype),'data_bytes':data_path.stat().st_size,'data_mtime_ns':data_path.stat().st_mtime_ns,
                'physical_gpu_index':0,'gpu_uuid':fields[0],'gpu_preflight_csv':query,'previous_cuda_visible_devices':previous_visibility,'effective_cuda_visible_devices':fields[0],
                'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'numpy_version':np.__version__,'python':sys.version,'python_executable':sys.executable,
                'optimizer':{'name':'Adam','betas':[0.9,0.999],'eps':1e-8,'weight_decay':0},
                'lr_schedule':'32**(-0.5)*min(global_step**(-0.5),global_step*warmups**(-1.5)); warmups=0 => 0.01',
                'initialization':'seed reset after construction; Xavier uniform Linear weights/zero biases; nn.Embedding normal(std=0.02); unknown zero',
                'shuffle':'NumPy default_rng(seed+1000003), full permutation each real epoch, retain short tail batches; stop exactly at total_steps',
                'deterministic_algorithms':True,'tf32':False,'precision':'float32','checkpoint_policy':'fixed final step; no early stopping or selected epoch; optimizer state not saved'}
        write_json(out/'config.json',config)
        sources=out/'sources';sources.mkdir();shutil.copy2(__file__,sources/'train_improvement.py');shutil.copy2(ROOT/'src/models.py',sources/'own_models.py');shutil.copy2(domain_path,out/'domains.json')
        if order_source:shutil.copy2(order_source,out/'column_order.json')
        torch.cuda.reset_peak_memory_stats(device);status.update(status='RUNNING',phase='training',total_steps=total_steps);write_json(out/'status.json',status)
        def on_step(record):
            if record['step']%args.log_every==0:
                status.update(steps_completed=record['step'],epoch_in_progress=record['real_epoch'],rows_in_epoch=record['rows_in_epoch'],rows_seen=record['rows_seen'],lr=record['lr'],updated_at=utcnow(),elapsed_seconds=time.monotonic()-started)
                write_json(out/'status.json',status);print(json.dumps({'event':'progress',**status}),flush=True)
        def on_window(record):
            record=dict(record,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            with (out/'epoch_metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            tmp=out/'checkpoint.pt.tmp';torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':config,'epochs_completed':record['real_epochs_completed'],'global_step':record['step'],'optimizer_state_saved':False},tmp);os.replace(tmp,out/'checkpoint.pt')
            status.update(epochs_completed=record['real_epochs_completed'],virtual_epochs_completed=record['virtual_epoch'],steps_completed=record['step'],rows_seen=record['rows_seen'],real_epoch=record['real_epoch'],real_epoch_progress=record['real_epoch_progress'],checkpoint='checkpoint.pt',last_epoch=record,updated_at=utcnow(),elapsed_seconds=time.monotonic()-started)
            write_json(out/'status.json',status);print(json.dumps({'event':'window_complete',**record}),flush=True)
        last=optimize(model,optimizer,table,order,device,args.seed,args.batch,total_steps,args.warmups,window_steps,on_step,on_window)
        if last['step']!=total_steps:raise AssertionError('incorrect final step')
        status.update(phase='final_nll',updated_at=utcnow());write_json(out/'status.json',status);model.eval();total_nll=0.0;eval_start=time.monotonic()
        with torch.no_grad():
            for begin in range(0,n,args.batch):
                raw=np.asarray(table[begin:begin+args.batch],dtype=np.int64)
                if order!=list(range(len(sizes))):raw=np.ascontiguousarray(raw[:,order])
                values=model.nll(torch.from_numpy(raw).to(device))
                if not bool(torch.isfinite(values).all()):raise FloatingPointError('nonfinite final NLL')
                total_nll+=values.double().sum().item()
        torch.cuda.synchronize(device)
        result={'final_full_table_nll_nats':total_nll/n,'evaluation_seconds':time.monotonic()-eval_start,'training_epochs':last['real_epoch_progress'],'real_epochs_completed':last['real_epochs_completed'],'real_epoch_in_progress':last['real_epoch'],'virtual_epochs':last['virtual_epoch'],'training_steps':total_steps,'rows':n,'training_rows_seen':last['rows_seen'],'checkpoint_bytes':(out/'checkpoint.pt').stat().st_size,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(device),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        write_json(out/'training_result.json',result);status.update(status='COMPLETE',phase='done',ended_at=utcnow(),exit_code=0,elapsed_seconds=time.monotonic()-started,result=result);write_json(out/'status.json',status);print(json.dumps({'event':'complete',**status}),flush=True);return 0
    except BaseException as error:
        status.update(status='FAILED',ended_at=utcnow(),exit_code=1,elapsed_seconds=time.monotonic()-started,error_type=type(error).__name__,error=str(error));(out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write_json(out/'status.json',status);print(json.dumps(status),file=sys.stderr,flush=True);return 1


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--purpose',choices=['benchmark','synthetic_check'],default='benchmark');p.add_argument('--encoded-dir',required=True);p.add_argument('--scale',choices=['1000000','5000000','full'],required=True);p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--run-dir',required=True);p.add_argument('--order-file');p.add_argument('--max-steps',type=int);p.add_argument('--batch',type=int,default=2048);p.add_argument('--width',type=int,default=256);p.add_argument('--layers',type=int,default=5);p.add_argument('--warmups',type=int,default=8000);p.add_argument('--threads',type=int,default=4);p.add_argument('--log-every',type=int,default=100);return p


if __name__=='__main__':
    sys.exit(main(parser().parse_args()))

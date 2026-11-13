"""Sequential frozen improvement queue; no training or torch import at import time.

Run on the authorized server after evidence/improvement_checks.json is PASS.
Failed or incomplete child directories are preserved and never overwritten.
"""
import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def require(value, message):
    if not value:
        raise AssertionError(message)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_embedding_mapping(cfg, cell):
    """Plan columns are a natural-order set; config lists follow model order."""
    order = cfg['column_order']
    require(sorted(order) == list(range(len(order))), 'Invalid model column permutation')
    expected_original = cell['embedding_original_columns']
    require(len(expected_original) == len(set(expected_original)), 'Duplicate planned embedding column')
    expected_positions = [pos for pos, original in enumerate(order) if original in set(expected_original)]
    require(cfg['embedding_columns'] == expected_positions, 'Embedding model positions mismatch')
    require(cfg['embedding_original_columns'] == [order[pos] for pos in expected_positions],
            'Embedding original/model coordinate mapping mismatch')
    require(set(cfg['embedding_original_columns']) == set(expected_original), 'Embedding selection mismatch')


def queries_for(cell):
    queries = [json.loads(s) for s in Path(cell['queries_path']).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(len(queries) == cell['expected_queries'], 'Frozen query count mismatch: '+cell['id'])
    require(len({q['query_id'] for q in queries}) == len(queries), 'Duplicate frozen query ID')
    require(all(q['workload'] == cell['scope'] and math.isfinite(q['truth']) and q['truth'] >= 0 for q in queries), 'Frozen workload/truth mismatch')
    return queries


def quantile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def basic_metrics(rows):
    errors = [float(r['qerror']) for r in rows]
    times = [float(r['latency_ms']) for r in rows]
    return dict(qerror_median=quantile(errors,.5),qerror_p95=quantile(errors,.95),
                qerror_p99=quantile(errors,.99),qerror_max=max(errors),latency_median_ms=quantile(times,.5))


def validate_evaluation(directory, cell, methods):
    """Validate one model slice of a potentially mixed CPU/model historical CSV."""
    directory = Path(directory)
    require(read_json(directory/'status.json')['status'] == 'COMPLETE', 'Incomplete evaluation: '+str(directory))
    summary = read_json(directory/'summary.json')
    require(summary.get('status','COMPLETE') == 'COMPLETE', 'Incomplete summary: '+str(directory))
    queries = queries_for(cell)
    with (directory/'per_query.csv').open(encoding='utf-8',newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        require(fields and {'query_id','workload','truth','method','estimate','qerror','latency_ms'} <= set(fields), 'CSV schema mismatch')
        raw = list(reader)
    selected = next((m for m in methods if any(r['method'] == m for r in raw)), None)
    require(selected is not None, 'Requested model absent from CSV: '+str(directory))
    rows = [r for r in raw if r['method'] == selected]
    require(len(rows) == len(queries), 'CSV model query count mismatch: '+str(directory))
    require([r['query_id'] for r in rows] == [str(q['query_id']) for q in queries], 'CSV query ID/order mismatch: '+str(directory))
    for row, q in zip(rows,queries):
        numbers = {k:float(row[k]) for k in ('truth','estimate','qerror','latency_ms')}
        require(all(math.isfinite(x) and x >= 0 for x in numbers.values()), 'Invalid CSV numbers')
        require(numbers['truth'] == q['truth'] and row['workload'] == q['workload'], 'CSV frozen truth/workload mismatch')
        truth, est = max(1,numbers['truth']), max(1,numbers['estimate'])
        require(math.isclose(numbers['qerror'],max(truth/est,est/truth),rel_tol=1e-8,abs_tol=1e-8), 'CSV q-error mismatch')
    require(summary.get('queries',len(queries)) == len(queries), 'Summary query count mismatch')
    stated = summary.get('methods',{}).get(selected,{})
    for key, value in basic_metrics(rows).items():
        if key in stated:
            require(math.isclose(value,stated[key],rel_tol=1e-8,abs_tol=1e-8), 'Summary/CSV metric mismatch: '+key)
    if 'training_seed' in summary:
        require(summary['training_seed'] == cell['seed'], 'Summary train seed mismatch')
    if 'scale' in summary:
        require(str(summary['scale']) == cell['scale'], 'Summary scale mismatch')
    if 'sampling_budget' in summary:
        require(summary['sampling_budget'] == cell['samples'], 'Summary sampling budget mismatch')
    if 'mc_seed' in summary:
        require(summary['mc_seed'] == cell['mc_seed'], 'Summary MC seed mismatch')
    return fields, rows, selected, summary


def materialize_reuse(source, destination, cell, method, alias_of=None):
    """Only change the selected CSV method; retain estimates, timings and strings."""
    source, destination = Path(source), Path(destination)
    fields, rows, selected, old_summary = validate_evaluation(source,cell,[method,'own_resmade'])
    destination.mkdir(parents=True,exist_ok=False)
    state = dict(status='STARTING',evaluation_performed=False,source_dir=str(source),
                 source_csv=str(source/'per_query.csv'),source_method=selected,
                 alias_of=alias_of,queries_completed=0,method=method)
    write_json(destination/'status.json',state)
    try:
        with (destination/'per_query.csv').open('x',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row,method=method))
        summary = dict(old_summary,status='COMPLETE',queries=len(rows),method=method,
                       variant=cell.get('variant','baseline'),scale=cell['scale'],
                       training_seed=cell['seed'],scope=cell['scope'],mc_seed=cell['mc_seed'],
                       sampling_budget=cell['samples'],evaluation_performed=False,
                       source_dir=str(source),source_csv=str(source/'per_query.csv'),
                       source_method=selected,alias_of=alias_of,
                       reused_from_baseline=bool(cell.get('reused_from_baseline',False)),
                       methods={method:old_summary.get('methods',{}).get(selected,basic_metrics(rows))})
        write_json(destination/'summary.json',summary)
        state.update(status='COMPLETE',queries_completed=len(rows),exit_code=0)
        write_json(destination/'status.json',state)
        validate_evaluation(destination,cell,[method])
    except BaseException as error:
        state.update(status='FAILED',error=repr(error),exit_code=1)
        write_json(destination/'status.json',state)
        (destination/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


class Queue:
    def __init__(self,args):
        self.args = args
        self.root = Path(args.experiment_root).resolve()
        self.project = self.root.parent.parent
        self.code = self.root/'code'
        self.encoded = Path(args.encoded_dir).resolve()
        self.plan = read_json(self.root/'plan.json')
        require(str(self.project) == self.plan['remote_project_root'], 'Run in the frozen remote project; plan paths are immutable')
        self.trains = {c['id']:c for c in self.plan['train_cells']}
        self.evals = {c['id']:c for c in self.plan['eval_cells']}
        self.state = dict(status='STARTING',pid=os.getpid(),started_at=utcnow(),
                          physical_gpu_index=0,active_pid=None,phase='preflight',active_id=None,
                          completed_train_ids=[],completed_eval_ids=[],completed_baseline_eval_ids=[],
                          new_fits_completed=0,reused_fits_completed=0,logical_evals_completed=0,
                          actual_new_inference_cells=0,actual_new_estimates=0,baseline_dev_reused=0,
                          planned_new_fits=33,planned_reused_fits=3,planned_logical_evals=720,
                          maximum_new_inference_cells=self.plan['new_inference_cells_max'],
                          maximum_new_estimates=self.plan['new_inference_rows_max'])
        self.initial_uuid = None
        (self.root/'queue_logs').mkdir(exist_ok=True)

    def save(self,**changes):
        self.state.update(changes,updated_at=utcnow())
        write_json(self.root/'queue_status.json',self.state)

    def gpu_guard(self):
        while True:
            raw = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
            fields = [s.strip() for s in raw.split(',')]
            require(len(fields)==4 and 'RTX 3090' in fields[1], 'Physical GPU0 is not the approved RTX3090: '+raw)
            if self.initial_uuid is None:
                self.initial_uuid = fields[0]
                approved = read_json(self.root/'evidence/improvement_checks.json')['gpu_uuid']
                require(self.initial_uuid == approved, 'GPU0 differs from the GPU used by correctness gates')
            require(fields[0] == self.initial_uuid, 'Physical GPU0 UUID changed during queue')
            apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
            occupied = [line.strip() for line in apps.splitlines() if line.split(',',1)[0].strip() == self.initial_uuid]
            busy = occupied or int(fields[2]) > 1000 or int(fields[3]) > 10
            self.save(status='WAITING' if busy else 'RUNNING',gpu_uuid=self.initial_uuid,
                      gpu_preflight=raw,gpu_compute_apps=occupied,active_pid=None)
            if not busy:
                return
            time.sleep(min(self.args.poll_seconds,60))

    def child(self,identifier,phase,script,arguments,gpu=True):
        self.save(phase=phase,active_id=identifier,active_pid=None)
        if gpu:
            self.gpu_guard()
        command = [sys.executable,str(self.code/'scripts'/script),*map(str,arguments)]
        require(Path(command[1]).is_file(), 'Missing frozen child script: '+command[1])
        log = self.root/'queue_logs'/(identifier+'.log')
        env = dict(os.environ,OMP_NUM_THREADS='4')
        env['PYTHONPATH'] = str(self.code)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        if gpu:
            env['CUDA_VISIBLE_DEVICES'] = self.initial_uuid
        with log.open('x',encoding='utf-8') as f:
            process = subprocess.Popen(command,cwd=self.project,stdout=f,stderr=subprocess.STDOUT,env=env)
            self.save(status='RUNNING',active_pid=process.pid,active_command=command,active_log=str(log))
            while process.poll() is None:
                time.sleep(min(self.args.poll_seconds,5))
                self.save()
            code = process.returncode
        self.save(active_pid=None,last_exit_code=code)
        require(code == 0,'Child failed; preserved directory/log: '+identifier)

    def validate_train(self,cell):
        out = Path(cell['run_dir'])
        require(read_json(out/'status.json')['status']=='COMPLETE','Training is incomplete: '+str(out))
        cfg = read_json(out/'config.json');result = read_json(out/'training_result.json')
        require((str(cfg['scale']),cfg['seed'],cfg['model']) == (cell['scale'],cell['seed'],'own_resmade'),'Train identity mismatch')
        require(all(cfg[k]==v for k,v in [('batch',2048),('width',256),('layers',5),('warmups',8000)]),'Training hyperparameter mismatch')
        require(all(cfg[k] is True for k in ('residual','direct_io','column_masking')),'ResMADE features differ from frozen baseline')
        require(cfg['deterministic_algorithms'] is True and cfg['tf32'] is False,'Training precision/determinism differs')
        require(cfg['parameter_count']>0 and cfg['parameter_bytes']>0,'Invalid parameter budget')
        require(cfg['rows']==cell['rows'] and result['rows']==cell['rows'],'Training row count mismatch')
        require(result['training_steps']==cell['total_steps'],'Training step budget mismatch')
        require(Path(cfg['data_path'])==self.encoded/(cell['scale']+'.npy'),'Training dataset differs from queue')
        require(Path(cfg['domain_path'])==self.encoded/(cell['scale']+'_domains.json'),'Training dictionary differs from queue')
        require((out/'checkpoint.pt').stat().st_size==result['checkpoint_bytes']>0,'Checkpoint missing or byte count mismatch')
        curve = [json.loads(s) for s in (out/'epoch_metrics.jsonl').read_text().splitlines() if s.strip()]
        require(len(curve)==20 and curve[-1]['global_step']==cell['total_steps'],'Training curve coverage mismatch')
        require(all(a['global_step']<b['global_step'] for a,b in zip(curve,curve[1:])),'Training steps are nonmonotone')
        require(all(math.isfinite(r['mean_training_nll_nats']) and r['rows']>0 and r['last_lr']>0 for r in curve),'Invalid training curve')
        require(math.isfinite(result['final_full_table_nll_nats']),'Nonfinite final NLL')
        if cell['reused_from_baseline']:
            require(cfg['input_encoding']=='binary' and cfg['column_order']==list(range(len(cfg['domain_sizes']))),'Reused B is not the frozen natural binary model')
        else:
            require(cfg['variant']==cell['variant'],'Training variant mismatch')
            validate_embedding_mapping(cfg,cell)
            expected_order = read_json(cell['order_file'])['order'] if cell['variant'] in ('order','all') else list(range(len(cfg['domain_sizes'])))
            require(cfg['column_order']==expected_order,'Model column order mismatch')
        return cfg

    def training(self,cell):
        out = Path(cell['run_dir'])
        if not out.exists():
            require(not cell['reused_from_baseline'],'Frozen B checkpoint is missing')
            arguments = ['--encoded-dir',self.encoded,'--scale',cell['scale'],'--variant',cell['variant'],
                         '--seed',cell['seed'],'--run-dir',out,'--max-steps',cell['total_steps']]
            if cell['variant'] in ('order','all'):
                arguments += ['--order-file',cell['order_file']]
            self.child(cell['id'],'training','train_improvement.py',arguments)
        self.validate_train(cell)
        self.state['completed_train_ids'].append(cell['id'])
        key = 'reused_fits_completed' if cell['reused_from_baseline'] else 'new_fits_completed'
        self.state[key] += 1
        self.save()

    def evaluation(self,cell,baseline=False,source=None):
        method = 'own_resmade' if baseline else 'own_resmade_'+cell['variant']
        out = Path(cell['output_dir'])
        if out.exists():
            validate_evaluation(out,cell,[method])
        else:
            if source is None and cell.get('alias_of'):
                source = self.evals[cell['alias_of']]['output_dir']
            if source is None and cell.get('reused_from_baseline'):
                source = cell['reuse_source_dir']
            if source:
                self.save(phase='materializing_reuse',active_id=cell['id'],active_pid=None)
                materialize_reuse(source,out,cell,method,cell.get('alias_of'))
            else:
                arguments = ['--run',cell['run_dir'],'--queries',cell['queries_path'],'--scope',cell['scope'],
                             '--output',out,'--samples',cell['samples'],'--mc-seed',cell['mc_seed']]
                if baseline:
                    arguments += ['--method','own_resmade']
                self.child(cell['id'],'baseline_dev' if baseline else 'evaluating','evaluate_improvement.py',arguments)
                validate_evaluation(out,cell,[method])
        summary = read_json(out/'summary.json')
        performed = summary.get('evaluation_performed',True)
        if not baseline:
            require(performed == (not cell.get('alias_of') and not cell.get('reused_from_baseline')),'Evaluation provenance mismatch')
            expected_source = (self.evals[cell['alias_of']]['output_dir'] if cell.get('alias_of')
                               else cell.get('reuse_source_dir'))
            if expected_source:
                require(Path(summary['source_dir'])==Path(expected_source),'Reused evaluation source differs from frozen plan')
        if performed:
            self.state['actual_new_inference_cells'] += 1
            self.state['actual_new_estimates'] += cell['expected_queries']
        if baseline:
            self.state['completed_baseline_eval_ids'].append(cell['id'])
            self.state['baseline_dev_reused'] += int(not performed)
        else:
            self.state['completed_eval_ids'].append(cell['id'])
            self.state['logical_evals_completed'] += 1
        self.save()

    def existing_baseline_dev(self,cell):
        """Inspect only experiments/*dev* and experiments/*/*dev* directories."""
        experiments = self.project/'experiments'
        candidates = sorted({p for pattern in ('*dev*','*/*dev*') for p in experiments.glob(pattern) if p.is_dir()})
        for candidate in candidates:
            if candidate == Path(cell['output_dir']):
                continue
            try:
                state = read_json(candidate/'status.json')
                arguments = state.get('arguments',{})
                # Exact run and query origins avoid confusing same-seed scales.
                if Path(arguments.get('run','__missing__')) != Path(cell['run_dir']) or Path(arguments.get('queries','__missing__')) != Path(cell['queries_path']):
                    continue
                if arguments.get('samples',2000) != 2000 or arguments.get('mc_seed',0) != 0:
                    continue
                validate_evaluation(candidate,cell,['own_resmade'])
                return candidate
            except (OSError,ValueError,KeyError,AssertionError,TypeError):
                continue
        return None

    def run(self):
        require(sys.platform=='linux','The experiment queue runs only on the authorized Linux server')
        require(read_json(self.root/'evidence/improvement_checks.json')['status']=='PASS','Merged preflight correctness checks must PASS')
        require(len(self.trains)==36 and len(self.evals)==720,'Frozen plan grid mismatch')
        require(sum(not c['reused_from_baseline'] for c in self.trains.values())==33,'Frozen fit count mismatch')
        for script in ('train_improvement.py','evaluate_improvement.py','run_improvement_matched.py','analyze_improvements.py'):
            require((self.code/'scripts'/script).is_file(),'Missing frozen code: '+script)
        spec = importlib.util.spec_from_file_location('frozen_queue_plan_auditor',self.code/'scripts/analyze_improvements.py')
        auditor = importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
        auditor.validate_plan(self.plan)
        self.save()
        for original in self.plan['baseline_eval_cells']:
            cell = dict(original,samples=2000,mc_seed=0,expected_queries=200)
            source = None if Path(cell['output_dir']).exists() else self.existing_baseline_dev(cell)
            self.evaluation(cell,baseline=True,source=source)
        for cell in self.plan['train_cells']:
            self.training(cell)
            for evaluation in self.plan['eval_cells']:
                if evaluation['train_id']==cell['id'] and evaluation['workload']!='budget':
                    self.evaluation(evaluation)
        for cell in self.plan['eval_cells']:
            if cell['workload']=='budget':
                self.evaluation(cell)
        matched = self.root/'matched'
        if matched.exists():
            require(read_json(matched/'status.json')['status']=='COMPLETE','Existing matched directory is not COMPLETE')
            manifest = read_json(matched/'manifest.json')
            require(len(manifest['draws'])==9,'Matched draw count mismatch')
            for draw in manifest['draws']:
                base = next(c for c in self.plan['eval_cells'] if c['scale']==draw['scale'] and c['workload']=='main_native')
                validate_evaluation(matched/draw['id'],base,['uniform_matched_improvement'])
        else:
            self.child('matched','matched_sampling','run_improvement_matched.py',['--experiment-root',self.root],gpu=False)
            require(read_json(matched/'status.json')['status']=='COMPLETE','Matched sampling did not complete')
        analysis = self.root/'analysis'
        if analysis.exists():
            require((analysis/'audit.json').is_file(),'Existing analysis directory has no complete audit')
            require(read_json(analysis/'audit.json')['status']=='PASS_COMPLETE','Existing analysis audit is not PASS_COMPLETE')
        else:
            self.child('analysis','analysis','analyze_improvements.py',['--experiment-root',self.root],gpu=False)
        require(read_json(analysis/'audit.json')['status']=='PASS_COMPLETE','Final analysis is incomplete')
        require(self.state['new_fits_completed']==33 and self.state['reused_fits_completed']==3 and self.state['logical_evals_completed']==720,'Queue completion count mismatch')
        new_baseline_dev = 8-self.state['baseline_dev_reused']
        require(self.state['actual_new_inference_cells']==627+new_baseline_dev and
                self.state['actual_new_estimates']==897600+200*new_baseline_dev,'Actual computation coverage mismatch')
        self.save(phase='final_gpu_idle',active_id=None)
        self.gpu_guard()
        self.save(status='COMPLETE',phase='done',active_id=None,active_pid=None,ended_at=utcnow())


def main(args):
    queue = None
    lock = Path(args.experiment_root).resolve()/'queue.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        queue = Queue(args)
        queue.run()
        return 0
    except BaseException as error:
        if queue is not None:
            queue.save(status='FAILED',error=repr(error),ended_at=utcnow())
            failure = queue.root/'queue_logs'/('failure_'+str(time.time_ns())+'.txt')
            failure.write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',required=True)
    parser.add_argument('--encoded-dir',required=True)
    parser.add_argument('--poll-seconds',type=int,default=60)
    args = parser.parse_args()
    require(args.poll_seconds>0,'Positive poll interval required')
    raise SystemExit(main(args))

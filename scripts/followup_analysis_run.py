"""Offline descriptive analyses over frozen results. No fitting or GPU access.

All outputs stay in experiments/followup_analysis_20261111. Raw inputs are read
only. Run with standard-library Python from the project workspace.
"""
import csv
import itertools
import json
import math
import statistics as st
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/followup_analysis_20261111'
SNAP = ROOT / 'data/snapshot_resume_20261109_172915'
SCALES = ['1000000', '5000000', 'full']
NROWS = dict(zip(SCALES, [1000000, 5000000, 12540479]))
MODELS = ['own_made', 'own_resmade', 'official_resmade']
SEEDS = [0, 1, 2]
TABLES = {}
CHECKS = Counter()
WORKLOADS = {}
SOURCE_FILES = {}


def load(path):
    path = Path(path)
    SOURCE_FILES[str(path.relative_to(ROOT))] = path.stat().st_size
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    path = Path(path)
    SOURCE_FILES[str(path.relative_to(ROOT))] = path.stat().st_size
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def local(remote):
    return ROOT / str(remote).split('DSC5003_Naru_20261109/', 1)[1]


def write(name, rows):
    TABLES[name] = len(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def qt(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    i = int(pos)
    return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (pos - i)


def stats(values):
    return dict(median=st.median(values), p95=qt(values, .95),
                p99=qt(values, .99), maximum=max(values))


def signature(q):
    return json.dumps([(p['column'], p['operator'], p['value']) for p in
                       sorted(q['predicates'], key=lambda p: p['column_index'])],
                      ensure_ascii=False)


def workload(scale, scope='main_native'):
    key = scale, scope
    if key not in WORKLOADS:
        if scope == 'main_native':
            p = SNAP / 'workloads' / scale / 'main_native.jsonl'
        else:
            root = 'workload_supplement_v2_20261110' if scope == 'stress_v2' else 'workload_supplement_20261110'
            p = ROOT / 'experiments' / root / scale / (scope + '.jsonl')
        SOURCE_FILES[str(p.relative_to(ROOT))] = p.stat().st_size
        qs = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
        assert len({q['query_id'] for q in qs}) == len(qs)
        for q in qs:
            q['_signature'] = signature(q)
        WORKLOADS[key] = {q['query_id']: q for q in qs}
    return WORKLOADS[key]


def validated_rows(path, scale, scope='main_native', method=None, pg=False):
    qs = workload(scale, scope)
    raw = records(path)
    rows = []
    seen = set()
    for r in raw:
        if method is not None and r['method'] != method:
            continue
        qid = r['query_id']
        assert qid not in seen
        seen.add(qid)
        t, e = float(r['truth']), float(r['estimate'])
        assert t == qs[qid]['truth'] and e >= 0 and math.isfinite(e)
        err = max(max(1, t) / max(1, e), max(1, e) / max(1, t))
        saved = r['q_error_floor1'] if pg else r['qerror']
        assert math.isclose(err, float(saved), rel_tol=1e-10)
        row = dict(query_id=qid, truth=t, estimate=e, qerror=err,
                   signature=qs[qid]['_signature'])
        if pg:
            row.update(planning_ms=float(r['planning_ms']), client_wall_ms=float(r['client_wall_ms']))
        else:
            row['latency_ms'] = float(r['latency_ms'])
        rows.append(row)
    assert len(rows) == len(qs), (path, len(rows), len(qs))
    CHECKS['raw_rows_reconstructed_for_analysis'] += len(rows)
    return rows


def semantic_errors(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['signature']].append(r['qerror'])
    return {key: st.median(v) for key, v in groups.items()}


def full_frequencies():
    cache = OUT / 'full_snapshot_value_frequencies.csv'
    receipt = OUT / 'full_snapshot_frequency_receipt.json'
    full = load(SNAP / 'encoded/full_domains.json')
    cols = full['columns']
    freq = {c: Counter() for c in cols}
    if cache.exists() and receipt.exists() and load(receipt)['status'] == 'COMPLETE':
        for r in records(cache):
            freq[r['column']][json.loads(r['value_json'])] = int(r['rows'])
        assert all(sum(v.values()) == NROWS['full'] for v in freq.values())
        return freq
    started = time.perf_counter()
    count = 0
    source = SNAP / 'dmv.csv'
    with source.open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        assert next(reader) == cols
        for row in reader:
            assert len(row) == len(cols)
            for c, value in zip(cols, row):
                freq[c][None if value == '__NARU_NULL__' else value] += 1
            count += 1
    assert count == NROWS['full']
    assert all(set(v) == set(domain) for v, domain in zip(freq.values(), full['domains']))
    rows = [dict(column=c, value_json=json.dumps(v, ensure_ascii=False), rows=n,
                 full_snapshot_fraction=n/count) for c in cols for v, n in freq[c].items()]
    write(cache.name, rows)
    receipt.write_text(json.dumps(dict(status='COMPLETE', rows=count, columns=len(cols),
                                      source=str(source.relative_to(ROOT)), source_bytes=source.stat().st_size,
                                      elapsed_seconds=time.perf_counter()-started,
                                      scope='Exact full snapshot frequencies only; not per-subset marginals'), indent=2))
    return freq


def rank_average(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            ranks[k] = (i + j - 1)/2 + 1
        i = j
    return ranks


def correlation(x, y):
    ax, ay = st.mean(x), st.mean(y)
    numerator = sum((a-ax)*(b-ay) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a-ax)**2 for a in x) * sum((b-ay)**2 for b in y))
    return numerator/denominator if denominator else None


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / 'status.json'
    status.write_text(json.dumps(dict(status='RUNNING', started_unix=time.time(),
                                      scope='Local offline descriptive analysis; no GPU, fitting or final figures')))
    for name, expected in [('main_evaluation_audit.json', 'PASS'),
                           ('supplement_numerical_audit.json', 'PASS_NUMERICAL_SUPPLEMENTS'),
                           ('pg_validation.json', 'PASS')]:
        assert load(ROOT / 'evidence' / name)['status'] == expected
    freq = full_frequencies()
    # Main ranking: query-record summaries remain the primary metrics. Dedup
    # diagnostics first aggregate repeated identical predicates within each run.
    main = {}
    main_metrics = {}
    for scale in SCALES:
        for model in MODELS:
            for seed in SEEDS:
                folder = ROOT / 'experiments/main_evaluation_20261110' / f'{scale}_{model}_seed{seed}'
                rows = validated_rows(folder/'per_query.csv', scale, method=model)
                main[scale, model, seed] = rows
                met = stats([r['qerror'] for r in rows])
                summary = load(folder/'summary.json')['methods'][model]
                for key in ['median', 'p95', 'p99']:
                    assert math.isclose(met[key], summary['qerror_'+key], rel_tol=1e-9)
                main_metrics[scale, model, seed] = met
    pair_rows, consistency = [], []
    for scale in SCALES:
        for left, right in itertools.combinations(MODELS, 2):
            ratios_by_semantic = defaultdict(list)
            wins = Counter()
            for seed in SEEDS:
                a, b = semantic_errors(main[scale,left,seed]), semantic_errors(main[scale,right,seed])
                assert set(a) == set(b)
                ratios = []
                for key in a:
                    ratio = a[key]/b[key]
                    ratios.append(ratio)
                    ratios_by_semantic[key].append(ratio)
                ties = sum(math.isclose(x, 1., rel_tol=1e-10, abs_tol=1e-10) for x in ratios)
                better = sum(x < 1 and not math.isclose(x, 1., rel_tol=1e-10, abs_tol=1e-10) for x in ratios)
                p = dict(scale=scale, left=left, right=right, training_seed=seed,
                         query_records=len(main[scale,left,seed]), unique_semantic_queries=len(a),
                         left_win_fraction=better/len(a), tie_fraction=ties/len(a),
                         right_win_fraction=(len(a)-better-ties)/len(a), median_qerror_ratio=st.median(ratios),
                         p95_difference_records=main_metrics[scale,left,seed]['p95']-main_metrics[scale,right,seed]['p95'],
                         p99_difference_records=main_metrics[scale,left,seed]['p99']-main_metrics[scale,right,seed]['p99'],
                         p95_difference_semantic_medians=qt(list(a.values()),.95)-qt(list(b.values()),.95))
                pair_rows.append(p)
                wins['left_lower_p95_seeds'] += p['p95_difference_records'] < 0
                wins['left_lower_p99_seeds'] += p['p99_difference_records'] < 0
            allleft = sum(all(x < 1-1e-10 for x in v) for v in ratios_by_semantic.values())
            allright = sum(all(x > 1+1e-10 for x in v) for v in ratios_by_semantic.values())
            n = len(ratios_by_semantic)
            consistency.append(dict(scale=scale,left=left,right=right,training_seeds=3,
                                    unique_semantic_queries=n,left_wins_all_seeds=allleft,
                                    right_wins_all_seeds=allright,mixed_or_tied=n-allleft-allright,
                                    left_wins_all_seeds_fraction=allleft/n,
                                    right_wins_all_seeds_fraction=allright/n,
                                    median_log_ratio_seed_range=st.median([max(map(math.log,v))-min(map(math.log,v)) for v in ratios_by_semantic.values()]),
                                    **wins))
    write('paired_rankings.csv',pair_rows)
    write('paired_seed_consistency.csv',consistency)

    # MC variation conditional on a checkpoint and descriptive variation between
    # the three checkpoint means. No random-effects/CI interpretation is made.
    mc_stats, mc_data = {}, {}
    conditional, between = [], []
    for scale in SCALES:
        for budget in [100,500,2000,8000]:
            for seed in SEEDS:
                grid = []
                for mc in SEEDS:
                    folder = ROOT/'experiments/gpu_supplement_20261110'/f'budget_{scale}_seed{seed}_s{budget}_mc{mc}'
                    rows = validated_rows(folder/'per_query.csv',scale,method='own_resmade')
                    met = stats([r['qerror'] for r in rows])
                    mc_stats[scale,budget,seed,mc] = met
                    # First collapse semantically repeated queries to avoid
                    # counting the same predicate as independent workload units.
                    sem = semantic_errors(rows)
                    mc_data[scale,budget,seed,mc] = sem
                    grid.append(met)
                keys = set(mc_data[scale,budget,seed,0])
                assert all(set(mc_data[scale,budget,seed,m]) == keys for m in SEEDS)
                log_sds = [st.pstdev([math.log(mc_data[scale,budget,seed,m][q]) for m in SEEDS]) for q in keys]
                conditional.append(dict(scale=scale,sampling_budget=budget,training_seed=seed,mc_repeats=3,
                                        unique_semantic_queries=len(keys),p95_mc_mean=st.mean(x['p95'] for x in grid),
                                        p95_mc_sd=st.pstdev(x['p95'] for x in grid),p95_mc_min=min(x['p95'] for x in grid),p95_mc_max=max(x['p95'] for x in grid),
                                        p99_mc_mean=st.mean(x['p99'] for x in grid),p99_mc_sd=st.pstdev(x['p99'] for x in grid),
                                        median_query_logq_mc_sd=st.median(log_sds),p95_query_logq_mc_sd=qt(log_sds,.95)))
            cond = [x for x in conditional if x['scale']==scale and x['sampling_budget']==budget]
            keys = set(mc_data[scale,budget,0,0])
            means_per_query = {q:[st.mean(math.log(mc_data[scale,budget,s,m][q]) for m in SEEDS) for s in SEEDS] for q in keys}
            between.append(dict(scale=scale,sampling_budget=budget,training_seeds=3,mc_repeats_per_seed=3,
                                p95_training_mean_sd=st.pstdev(x['p95_mc_mean'] for x in cond),
                                mean_within_training_p95_mc_sd=st.mean(x['p95_mc_sd'] for x in cond),
                                p99_training_mean_sd=st.pstdev(x['p99_mc_mean'] for x in cond),
                                mean_within_training_p99_mc_sd=st.mean(x['p99_mc_sd'] for x in cond),
                                median_query_logq_training_mean_sd=st.median(st.pstdev(v) for v in means_per_query.values()),
                                median_conditional_query_logq_mc_sd=st.median(x['median_query_logq_mc_sd'] for x in cond)))
    write('mc_spread_by_training_seed.csv',conditional)
    write('training_vs_mc_spread.csv',between)

    # Group and tail cases. Full-snapshot equality-value frequency is a fixed
    # reference covariate. It is NOT an exact marginal for the smaller subsets.
    def features(q,scale):
        eq = [(p['column'],p['value']) for p in q['predicates'] if p['operator']=='=' and p['value'] is not None]
        minimum = min((freq[c][v]/NROWS['full'] for c,v in eq),default=None)
        rare_group = 'no_nonnull_equality' if minimum is None else '<=1e-5' if minimum<=1e-5 else '<=1e-4' if minimum<=1e-4 else '<=1e-3' if minimum<=1e-3 else '>1e-3'
        sel = q['truth']/NROWS[scale]
        sb = 'zero' if sel==0 else '<=1e-5' if sel<=1e-5 else '<=1e-3' if sel<=1e-3 else '<=1e-2' if sel<=1e-2 else '>1e-2'
        return dict(predicate_count=len(q['predicates']),selectivity_group=sb,
                    min_eq_full_snapshot_fraction=minimum,rare_equality_group=rare_group,
                    positive_truth_le10=0<q['truth']<=10,
                    has_date_equality=any(c=='reg_valid_date' for c,v in eq),
                    has_indicator_Y=any(c.endswith('_indicator') and v=='Y' for c,v in eq))
    grouped, tail_cases = [], []
    for scale in SCALES:
        for model in MODELS:
            for seed in SEEDS:
                qs = workload(scale)
                rows = main[scale,model,seed]
                groups = defaultdict(list)
                for row in rows:
                    fs = features(qs[row['query_id']],scale)
                    for key in ['predicate_count','selectivity_group','rare_equality_group','positive_truth_le10','has_date_equality','has_indicator_Y']:
                        groups[key,str(fs[key])].append(row)
                for (key,value),v in groups.items():
                    sem = semantic_errors(v)
                    grouped.append(dict(scale=scale,scope='main_native',model=model,training_seed=seed,grouping=key,group=value,
                                        query_records=len(v),unique_semantic_queries=len(sem),
                                        **{'qerror_'+k:v for k,v in stats([r['qerror'] for r in v]).items()},
                                        semantic_median_qerror_p95=qt(list(sem.values()),.95),
                                        underestimate_fraction=sum(r['estimate']<r['truth'] for r in v)/len(v)))
            # Fixed ranking: top ten semantic conditions by median of three
            # seed errors; report ALL seeds for every selected condition.
            qs = workload(scale)
            by_semantic = defaultdict(list)
            for seed in SEEDS:
                for row in main[scale,model,seed]:
                    by_semantic[row['signature']].append((seed,row))
            ranked = sorted(by_semantic, key=lambda key:(-st.median(r['qerror'] for s,r in by_semantic[key]),key))[:10]
            for rank,key in enumerate(ranked,1):
                vals = by_semantic[key]
                first = qs[vals[0][1]['query_id']]
                per_seed = {s:st.median(r['qerror'] for ss,r in vals if ss==s) for s in SEEDS}
                tail_cases.append(dict(scale=scale,model=model,rank=rank,selection='top10 median qerror across all three training seeds; semantic duplicates collapsed',
                                       representative_query_id=first['query_id'],truth=first['truth'],
                                       qerror_seed0=per_seed[0],qerror_seed1=per_seed[1],qerror_seed2=per_seed[2],
                                       qerror_seed_median=st.median(per_seed.values()),
                                       estimate_seed0=st.median(r['estimate'] for s,r in vals if s==0),
                                       estimate_seed1=st.median(r['estimate'] for s,r in vals if s==1),
                                       estimate_seed2=st.median(r['estimate'] for s,r in vals if s==2),
                                       **features(first,scale),predicates=key))
    write('tail_groups.csv',grouped)
    write('tail_cases_top10.csv',tail_cases)
    # Read the supplemental predicate scan independently (all three model
    # training seeds); only one and eleven predicates are needed for the
    # planned contrast, and their positive/zero sizes remain explicit.
    scan = []
    for scale in SCALES:
        qs = workload(scale,'predicate_sweep')
        for model in MODELS:
            for seed in SEEDS:
                folder=ROOT/'experiments/workload_eval_supplement_v2_20261110'/f'{scale}_{model}_seed{seed}_predicate_sweep'
                rows=validated_rows(folder/'per_query.csv',scale,'predicate_sweep',method=model)
                for count in [1,11]:
                    group=[r for r in rows if len(qs[r['query_id']]['predicates'])==count]
                    positive=[r for r in group if r['truth']>0]
                    scan.append(dict(scale=scale,model=model,training_seed=seed,predicate_count=count,query_records=len(group),
                                     positive_query_records=len(positive),unique_positive_semantics=len(semantic_errors(positive)),
                                     **{'positive_qerror_'+k:v for k,v in stats([r['qerror'] for r in positive]).items()}))
    write('predicate_extremes.csv',scan)
    zero = records(ROOT/'experiments/supplement_analysis_20261110/nontrivial_zero_estimates.csv')
    zgroups=defaultdict(list)
    for row in zero:zgroups[row['scale'],row['method']].append(row)
    zsummary=[]
    for (scale,method),v in zgroups.items():
        est=[float(r['estimate']) for r in v]
        zsummary.append(dict(scale=scale,method=method,estimate_records=len(v),distinct_query_ids=len({r['query_id'] for r in v}),
                             distinct_semantic_queries=len({json.dumps([(p['column'],p['operator'],p['value']) for p in sorted(json.loads(r['predicates']),key=lambda p:p['column_index'])]) for r in v}),
                             median_estimated_rows=st.median(est),max_estimated_rows=max(est),
                             estimate_gt1_fraction=sum(x>1 for x in est)/len(est),estimate_gt10_fraction=sum(x>10 for x in est)/len(est)))
    write('nontrivial_zero_summary.csv',zsummary)

    # Cost points preserve each method's measurement boundary. No global
    # end-to-end speedup or physical-storage Pareto ranking is computed.
    matrix=load(ROOT/'experiments/main_matrix_20261109/status.json')['runs']
    training, cost, nll_points=[],[],[]
    paths=[local(c['path']) for c in matrix]
    paths += [ROOT/'experiments/size_supplement_20261110'/f'width{w}_seed{s}' for w in [128,512] for s in SEEDS]
    for path in paths:
        cfg=load(path/'config.json');res=load(path/'training_result.json');state=load(path/'status.json')
        epochs=[json.loads(x) for x in (path/'epoch_metrics.jsonl').read_text().splitlines()]
        assert len(epochs)==20 and [x['epoch'] for x in epochs]==list(range(1,21)) and state['status']=='COMPLETE'
        y=[x['mean_training_nll_nats'] for x in epochs[-5:]]
        slope=sum((i-2)*(v-st.mean(y)) for i,v in enumerate(y))/10
        p=dict(scale=cfg['scale'],model=cfg['model'],width=cfg['width'],training_seed=cfg['seed'],
               training_steps=res['training_steps'],warmup_steps=cfg['warmups'],
               warmup_fraction_of_steps=cfg['warmups']/res['training_steps'],
               warmup_end_epoch=cfg['warmups']/cfg['batches_per_epoch'],
               loss_epoch15=epochs[14]['mean_training_nll_nats'],loss_epoch16=y[0],loss_epoch20=y[-1],
               loss_epoch16_to20_drop=y[0]-y[-1],loss_epoch15_to20_drop=epochs[14]['mean_training_nll_nats']-y[-1],
               last5_loss_slope_per_epoch=slope,final_unmasked_nll=res['final_full_table_nll_nats'],
               receipt_elapsed_seconds=state['elapsed_seconds'],epoch_seconds_sum=sum(x['elapsed_seconds'] for x in epochs),
               parameter_bytes=cfg['parameter_bytes'],checkpoint_bytes=res['checkpoint_bytes'],
               peak_gpu_allocated_bytes=res['peak_gpu_allocated_bytes'],
               training_loss_objective='column masking enabled' if cfg['column_masking'] else 'unmasked')
        training.append(p)
        scale,model,seed=cfg['scale'],cfg['model'],cfg['seed']
        if cfg['width']==256:
            data=main[scale,model,seed]
        else:
            data=validated_rows(ROOT/'experiments/gpu_supplement_20261110'/f"size_width{cfg['width']}_seed{seed}"/'per_query.csv',scale,method=model)
        ms=stats([r['qerror'] for r in data])
        point=dict(scale=scale,method=model,repeat_kind='training_seed',repeat=seed,width=cfg['width'],
                   query_records=len(data),**{'qerror_'+k:v for k,v in ms.items()},
                   inference_median_ms=st.median(r['latency_ms'] for r in data),
                   latency_boundary='prebuilt host masks to GPU + synchronized shared progressive sampler; excludes host numpy-mask construction',
                   build_seconds=state['elapsed_seconds'],build_boundary='20-epoch training receipt, including final NLL/checkpoint overhead',
                   logical_synopsis_bytes=cfg['parameter_bytes'],storage_boundary='FP32 trainable parameters; common dictionaries excluded',
                   checkpoint_file_bytes=res['checkpoint_bytes'],peak_gpu_allocated_bytes=res['peak_gpu_allocated_bytes'])
        cost.append(point);nll_points.append(dict(scale=scale,model=model,width=cfg['width'],training_seed=seed,
                                                 final_unmasked_nll=res['final_full_table_nll_nats'],qerror_p95=ms['p95'],qerror_p99=ms['p99']))
    write('convergence_by_run.csv',training)
    write('nll_query_points.csv',nll_points)
    for scale in SCALES:
        source=ROOT/'experiments/main_evaluation_20261110'/f'{scale}_own_resmade_seed0/per_query.csv'
        for method in ['exact_marginal_independence','uniform_sample']:
            data=validated_rows(source,scale,method=method)
            cfg=load(local(next(c['path'] for c in matrix if c['scale']==scale and c['model']=='own_resmade' and c['seed']==0))/'config.json')
            storage=8*sum(cfg['domain_sizes']) if method=='exact_marginal_independence' else 440000
            cost.append(dict(scale=scale,method=method,repeat_kind='fixed_baseline',repeat=20261109 if method=='uniform_sample' else '',width='',query_records=len(data),
                             **{'qerror_'+k:v for k,v in stats([r['qerror'] for r in data]).items()},
                             inference_median_ms=st.median(r['latency_ms'] for r in data),latency_boundary='CPU prebuilt synopsis lookup; setup excluded; see raw implementation for mask construction',
                             build_seconds=None,build_boundary='not measured',logical_synopsis_bytes=storage,
                             storage_boundary='int64 exact marginal counts, common dictionaries excluded' if method=='exact_marginal_independence' else '10000 materialized int32 rows, common dictionaries excluded'))
        for model in MODELS:
            for seed in SEEDS:
                folder=ROOT/'experiments/matched_sampling_20261110'/f'{scale}_{model}_sample_seed{seed}'
                s=load(folder/'summary.json');data=validated_rows(folder/'per_query.csv',scale)
                assert s['sample_array_bytes']==s['sample_rows']*44 and 0<=s['parameter_budget_bytes']-s['sample_array_bytes']<44
                cost.append(dict(scale=scale,method='uniform_matched_'+model,repeat_kind='sample_seed',repeat=seed,width='',query_records=len(data),
                                 **{'qerror_'+k:v for k,v in stats([r['qerror'] for r in data]).items()},
                                 inference_median_ms=st.median(r['latency_ms'] for r in data),latency_boundary='CPU sample scan; host mask construction excluded',
                                 build_seconds=None,build_boundary='not measured',logical_synopsis_bytes=s['sample_array_bytes'],
                                 storage_boundary='materialized int32 row bytes; common dictionaries excluded',sample_rows=s['sample_rows'],
                                 reference_parameter_budget_bytes=s['parameter_budget_bytes'],
                                 duplicate_budget_note='own/official ResMADE same row budget and seed means identical sample, not independent draws'))
        for tier in ['PG-default','PG-target1000','PG-extstats']:
            for rep in [1,2,3]:
                folder=ROOT/'experiments/pg_supplement_20261110'/scale/tier/f'repeat{rep}'
                data=validated_rows(folder/'main_native_metrics.csv',scale,pg=True);s=load(folder/'analyze.json')
                cost.append(dict(scale=scale,method=tier,repeat_kind='ANALYZE_repetition',repeat=rep,width='',query_records=len(data),
                                 **{'qerror_'+k:v for k,v in stats([r['qerror'] for r in data]).items()},
                                 inference_median_ms=None,latency_boundary='server EXPLAIN planning + client roundtrip separately; SQL not executed',
                                 planning_median_ms=st.median(r['planning_ms'] for r in data),client_wall_median_ms=st.median(r['client_wall_ms'] for r in data),
                                 build_seconds=s['analyze_seconds'],build_boundary='ANALYZE only; excludes data import and source build',
                                 logical_synopsis_bytes=s['single_column_catalog_row_bytes']+s['extended_catalog_row_bytes'],
                                 storage_boundary='pg_column_size attributed catalog rows; not physical pages/indexes/WAL/heap/executables',
                                 single_column_catalog_row_bytes=s['single_column_catalog_row_bytes'],extended_catalog_row_bytes=s['extended_catalog_row_bytes']))
    write('accuracy_cost_by_run.csv',cost)
    one=[x for x in nll_points if x['scale']=='1000000' and x['model']=='own_resmade']
    association=[]
    assert len(one)==9
    for target in ['qerror_p95','qerror_p99']:
        association.append(dict(scale='1000000',model='own_resmade',fits=9,widths='128/256/512',
                                target=target,spearman_nll_vs_query=correlation(rank_average([x['final_unmasked_nll'] for x in one]),rank_average([x[target] for x in one])),
                                interpretation='Descriptive only; width and seed vary, n=9, no causal or significance inference'))
    write('nll_query_association.csv',association)
    report=dict(status='PASS',completed_unix=time.time(),scope='Four offline analyses; no model fitting, GPU access, or final figures',
                input_bytes_by_path=SOURCE_FILES,checks=dict(CHECKS),tables=TABLES,
                limits=['Only three training and MC seeds; descriptive spreads, no confidence-interval or random-effects claim.',
                        'Semantic duplicates collapsed for paired/conditional diagnostics; original primary record-weighted quantiles remain unchanged.',
                        'Full-snapshot rare-value frequencies are reference covariates, not exact subset marginals.',
                        'Model, PG and sampling cost boundaries are distinct; no end-to-end speedup or physical-storage Pareto claim.',
                        'Masked training NLL, unmasked training NLL and final unmasked NLL are different objectives; no validation curve exists.'])
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    render_analysis(pair_rows,consistency,conditional,between,training,nll_points,cost,tail_cases,zsummary,association)
    status.write_text(json.dumps(dict(status='COMPLETE',ended_unix=time.time(),tables=TABLES,
                                      audit='audit.json',analysis='ANALYSIS.md'),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',tables=TABLES,checks=dict(CHECKS))))


def render_analysis(pairs,consistency,conditional,between,training,nll_points,cost,tail,zsummary,association):
    def fmt(x):return f'{x:.4f}'
    lines=['# 既有实验的四项离线分析', '',
           '本分析只读取已冻结的本地结果，不训练模型、不访问GPU、不改原始文件，也未制作最终图表。重建命令：', '',
           '```powershell', 'python Naru_Lab_20261109/scripts/followup_analysis_run.py', '```', '',
           '所有表均在本目录；audit.json记录输入路径、读取字节数、重算检查与行数。首次运行会流式扫描本地冻结DMV CSV，缓存精确full快照值频率；后续运行复用缓存。', '',
           '**配对排名与稳定性。** 主负载每规模2000条查询记录，三个训练seed。查询条件按列排序后定义语义身份；重复条件先在每run内部取q-error中位数，再计算配对胜率，避免将相同谓词的重复MC估计计作独立查询。主表的原记录加权p95/p99保持不变，并在paired_rankings.csv中并列报告。', '',
           '|规模|左方法 / 右方法|左方法胜率（三seed范围）|左方法p95较低的seed数|左方法p99较低的seed数|',
           '|---|---|---:|---:|---:|']
    for scale in SCALES:
        for left,right in itertools.combinations(MODELS,2):
            v=[p for p in pairs if p['scale']==scale and p['left']==left and p['right']==right]
            c=next(c for c in consistency if c['scale']==scale and c['left']==left and c['right']==right)
            wins=[p['left_win_fraction'] for p in v]
            lines.append(f"|{scale}|{left} / {right}|{min(wins):.1%}–{max(wins):.1%}|{c['left_lower_p95_seeds']}/3|{c['left_lower_p99_seeds']}/3|")
    lines += ['', '这里的胜率只描述保存查询上的误差排序，并不证明架构显著优越；MADE与ResMADE还同时存在masking/direct-io等功能差别。MADE及官方ResMADE主表只有MC seed0，无法从其三个训练seed中独立分离MC误差。paired_seed_consistency.csv给出同一语义条件在三个seed中持续胜出或发生反转的数量。', '',
              '**MC与训练波动。** mc_spread_by_training_seed.csv保留每个固定checkpoint的三个MC重复范围；training_vs_mc_spread.csv比较三个checkpoint的MC均值差异和各checkpoint内部MC差异。以下均为描述性总体标准差，只有3×3网格，不是误差方差的严格因果分解或95%CI。', '',
              '|规模|samples|p95训练均值之间SD|平均checkpoint内MC SD|语义query log(q-error)的MC SD中位数|',
              '|---|---:|---:|---:|---:|']
    for p in between:
        lines.append(f"|{p['scale']}|{p['sampling_budget']}|{fmt(p['p95_training_mean_sd'])}|{fmt(p['mean_within_training_p95_mc_sd'])}|{fmt(p['median_conditional_query_logq_mc_sd'])}|")
    lines += ['', '**尾部与失败模式。** tail_groups.csv按谓词数量、精确真值选择率、低基数(1–10行)、日期等值、Y标志及等值常量的full快照频率分组。稀有常量阈值在分析前固定为1e-5/1e-4/1e-3；它不等同于范围谓词选择率，且1M/5M使用的是full快照参照频率。每行保留记录数和语义去重数，小组的p95/p99不能视为稳健总体估计。', '',
              'tail_cases_top10.csv固定选择每规模/方法三个seed中位q-error最高的10种语义，并同时保留三个seed估计、真值和原值谓词。它适合解释失败案例，不用来调参或宣称总体性能。单谓词/11谓词对照在predicate_extremes.csv，分别列正结果数量。', '',
              '|规模|模型|最差语义的三seed中位q-error|真值|代表query ID|',
              '|---|---|---:|---:|---|']
    for scale in SCALES:
        for model in MODELS:
            p=next(p for p in tail if p['scale']==scale and p['model']==model and p['rank']==1)
            lines.append(f"|{scale}|{model}|{p['qerror_seed_median']:.3f}|{p['truth']}|{p['representative_query_id']}|")
    lines += ['', 'nontrivial_zero_summary.csv来自已有522条跨方法估计，按语义条件去重列出样本量和估计行数阈值。1M有14条、5M有4条查询记录，跨方法/seed重复不增加独立query数量。stress_v2的125条零结果全部存在空谓词域，只是语义一致性覆盖；这里不据此声称学会联合零结果。新联合零/近零实验由主线程另行负责，不混入旧结论。', '',
              '**精度与成本。** accuracy_cost_by_run.csv把模型参数、checkpoint文件、采样行数据、PG逻辑catalog行字节分列；共同词典在模型/采样预算中同时排除。own/official ResMADE的相同预算与样本seed会重复同一份样本，不能当独立抽样。以下full主负载取各重复的p95及对应存储中位数，存储边界不同，因此只给并列表，不给物理存储Pareto排名。', '',
              '|方法|p95中位数|报告的逻辑synopsis字节中位数|成本口径|',
              '|---|---:|---:|---|']
    for method in ['own_made','own_resmade','official_resmade','uniform_sample','uniform_matched_own_resmade','PG-default','PG-target1000','PG-extstats']:
        v=[x for x in cost if x['scale']=='full' and x['method']==method and (x['width']==256 or x['width']=='')]
        lines.append(f"|{method}|{st.median(x['qerror_p95'] for x in v):.4f}|{st.median(x['logical_synopsis_bytes'] for x in v):.0f}|{v[0]['storage_boundary']}|")
    lines += ['', 'GPU列是共享未优化sampler的同步单query推断，含预建host mask到GPU的转换/传输，不含构造host numpy mask。PG列分别保存服务器规划和客户端往返，未执行SQL；不可计算二者的端到端SQL加速比。模型构建列用训练回执，PG用ANALYZE，采样/边际构建时间原实验未测量，留空而不补造。PG源代码构建、数据导入及共同数据表未并入统计构建成本。', '',
              '**收敛、warmup与NLL。** convergence_by_run.csv保留33次正式拟合的每轮目标定义、epoch15/16到20的loss变化、末5轮线性趋势、warmup步数及占比。主矩阵20epochs在三个规模分别对应9780/48840/122480步，8000步warmup约占81.8%/16.4%/6.5%；规模效应仍混入优化预算。', '',
              '|规模|模型|epoch16→20 loss下降（三seed均值）|末5轮斜率均值|warmup结束epoch|',
              '|---|---|---:|---:|---:|']
    for scale in SCALES:
        for model in MODELS:
            v=[x for x in training if x['scale']==scale and x['model']==model and x['width']==256]
            lines.append(f"|{scale}|{model}|{st.mean(x['loss_epoch16_to20_drop'] for x in v):.5f}|{st.mean(x['last5_loss_slope_per_epoch'] for x in v):.5f}|{v[0]['warmup_end_epoch']:.2f}|")
    lines += ['', '1M末轮仍改善，缺少独立validation曲线，不能称充分收敛，也不能把训练loss下降解释为泛化改善。ResMADE每epoch loss含column masking，MADE不含；最终NLL是在同一训练表关闭masking计算，不能用它补成不存在的validation曲线。', '',
              'full自写ResMADE末段masked loss略上行；随机mask训练目标和缺少独立validation限制了解释，单凭该段不能判断过拟合或数值发散。', '',
              'nll_query_points.csv供后续散点图使用；1M ResMADE的9个拟合（3宽×3seed）中，NLL与查询分位数的Spearman仅作描述：']
    for p in association:lines.append(f"- {p['target']}：{p['spearman_nll_vs_query']:.4f}，n=9；width与seed同时变化，没有显著性/因果结论。")
    lines += ['', '当前分析支持继续报告反直觉结果：更低NLL不保证每个查询分位数更低，更多MC采样也不保证每次p99单调下降。不需要为统一模型排名而重训。若报告要把规模收益归因于样本量，才需要单独固定优化步数；若要宣称MADE优于ResMADE，则需要更多推断重复或更谨慎的条件性表述。', '',
              '原始输入入口：../main_evaluation_20261110/、../main_matrix_20261109/、../gpu_supplement_20261110/、../pg_supplement_20261110/、../matched_sampling_20261110/及数据快照workloads（项目根data/，完整路径见audit.json）。已有结果全部只读；本轮没有作图或生成report。']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'status.json').write_text(json.dumps(dict(status='FAILED',error=repr(exc),ended_unix=time.time()),indent=2))
        (OUT/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

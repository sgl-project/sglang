#!/usr/bin/env python3
"""Offline packaging/analysis only. No service, GPU, or control requests."""
import argparse, collections, csv, gzip, hashlib, io, json, math, random, shutil, statistics, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/sglang-rfc-build')
RUNS = ['bench-p2-matrix-v1','bench-p2-ready-window','bench-optimized-matrix-v1','bench-optimized-retention-matrix','bench-optimized-background-v1']
OPTIONAL = 'bench-optimized-host-pressure-v1'
OLD = '69de4c3e82494d3aa278b22a067bcfd901cda119'
NEW = '630da396a576b6bb1dceadbeb0f3442fc33c9f20'
NEG = {'unused','missing','cancel','route_change'}

def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
def gz(data):
    stream=io.BytesIO()
    with gzip.GzipFile(fileobj=stream,filename='',mode='wb',mtime=0,compresslevel=9) as f: f.write(data)
    return stream.getvalue()
def git(*args): return subprocess.check_output(['git','-C',str(BASE/'runtime-worktree'),*args])
def quantile(values,p):
    if not values:return None
    values=sorted(values); i=(len(values)-1)*p;a=math.floor(i);b=math.ceil(i)
    return values[a]+(values[b]-values[a])*(i-a)
def stable_seed(label): return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')
def bootstrap(values,label):
    if not values:return None
    seed=stable_seed(label);r=random.Random(seed)
    samples=[statistics.mean(r.choices(values,k=len(values))) for _ in range(4000)]
    return {'n_pairs':len(values),'mean':statistics.mean(values),'ci':[quantile(samples,.025),quantile(samples,.975)],'seed':seed,'resamples':4000,'confidence':.95,'method':'percentile bootstrap of paired mean deltas','small_sample':len(values)<20}
def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def trials_for(name):
    p=ROOT/name/'raw.jsonl.gz'
    return [v for line in gzip.decompress(p.read_bytes()).decode().splitlines() if (v:=json.loads(line)).get('kind')=='trial']

def package(names):
    inventory=[]
    for name in names:
        source=BASE/name;dest=ROOT/name;dest.mkdir(exist_ok=True)
        files={p.name:p.read_bytes() for p in source.iterdir() if p.name in ['manifest.json','schedule.json','summary.json','summary.md','prefetch-cost-ema.json','raw.jsonl']}
        summary=json.loads(files['summary.json']);schedule=json.loads(files['schedule.json']);lines=[json.loads(s) for s in files['raw.jsonl'].splitlines()];trials=[v for v in lines if v.get('kind')=='trial']
        if len(trials)!=len(schedule) or summary['attempts']!=len(schedule):raise RuntimeError(f'{name} is incomplete: trials={len(trials)} schedule={len(schedule)} summary={summary["attempts"]}')
        if summary['valid_attempts'] != sum(t.get('valid',False) for t in trials):raise RuntimeError('valid attempt count mismatch')
        if {x['trial'] for x in summary['failures']} != {t['id'] for t in trials if not t.get('valid')}:raise RuntimeError('failure set mismatch')
        ids=[t['id'] for t in trials]
        if len(set(ids))!=len(ids) or set(ids)!={x['id'] for x in schedule}:raise RuntimeError('trial/schedule mismatch')
        for filename,data in files.items(): (dest/(filename+'.gz' if filename=='raw.jsonl' else filename)).write_bytes(gz(data) if filename=='raw.jsonl' else data)
        # Recheck the source after packaging; never silently freeze a live writer.
        if any((source/k).read_bytes()!=v for k,v in files.items()):raise RuntimeError(f'{name} changed while copying')
        inventory.append({'run':name,'attempts':len(trials),'valid_attempts':summary['valid_attempts'],'failures':summary['failures'],'raw_jsonl_sha256':sha(files['raw.jsonl']),'raw_jsonl_bytes':len(files['raw.jsonl']),'raw_jsonl_lines':len(lines),'raw_gzip_sha256':sha((dest/'raw.jsonl.gz').read_bytes()),'raw_gzip_mtime':0,'raw_gzip_filename':'','runtime_commit':OLD if name.startswith('bench-p2-') else NEW,'raw_first_wall_time':lines[0].get('wall_time'),'raw_last_wall_time':lines[-1].get('wall_time')})
    dump(ROOT/'runs.json',inventory)
    evidence=ROOT/'provenance';evidence.mkdir(exist_ok=True)
    (evidence/'start-feature-engine.py').write_bytes((BASE/'start-feature-engine.py').read_bytes())
    (evidence/'runtime-reflog.txt').write_bytes(git('reflog','--date=iso','-6','HEAD'))
    for name in ['p2-original-engine-a.log','p2-original-engine-b.log']:(evidence/(name+'.gz')).write_bytes(gz((BASE/name).read_bytes()))
    files=['python/sglang/srt/mem_cache/unified_cache/control_storage_reads.py','python/sglang/srt/mem_cache/unified_cache/control_storage.py','python/sglang/srt/mem_cache/unified_cache/storage_execution.py','python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py','python/sglang/srt/mem_cache/unified_cache/kv_control_adapter.py']
    snapshots={}
    for commit in [OLD,NEW]:
        hashes={}
        for path in files:
            try:hashes[path]=sha(git('show',f'{commit}:{path}'))
            except subprocess.CalledProcessError:hashes[path]=None
        snapshots[commit]={'tree':git('rev-parse',f'{commit}^{{tree}}').decode().strip(),'subject':git('show','-s','--format=%s',commit).decode().strip(),'source_files_sha256':hashes}
    dump(evidence/'runtime-provenance.json',{'old_runtime_commit':OLD,'old_snapshot_scope':'P0-P3 snapshot; directory name p2 is historical; no L2 PIN','optimized_runtime_commit':NEW,'optimized_snapshot_scope':'includes P4/P5 plus Engine changes; before/after is not isolated optimization','runtime_worktree_head_at_packaging':git('rev-parse','HEAD').decode().strip(),'runtime_history_evidence':'runtime-reflog.txt','source_launcher':'start-feature-engine.py','source_launcher_uses':'PYTHONPATH /tmp/sglang-rfc-build/runtime-worktree/python','run_assignment_basis':'parent verified runtime worktree creation/checkout reflog and launcher; original benchmark manifests do not record runtime commit','manifest_git_warning':'A benchmark source Git HEAD, when supplied by another manifest, identifies the harness/repository state, not automatically the loaded Engine runtime snapshot. Original copied manifests contain no source_git field.','snapshots':snapshots,'cross_run_confounds':['runtime snapshot includes additional P4/P5 changes','harness script/modules changed','N differs (short window 6 vs 10; other runs 3 or 4)','separate executions and cache/host scheduling epochs']})
    harness=evidence/'harness-at-packaging';harness.mkdir(exist_ok=True)
    current={}
    for p in sorted((BASE/'benchmark-work').glob('*.py')):
        if p.name not in ['session_kv_bench.py','bench_statistics.py','bench_report.py','bench_inputs.py','bench_load.py']:continue
        data=p.read_bytes();(harness/p.name).write_bytes(data);current[p.name]=sha(data)
    dump(evidence/'harness-hashes.json',{'copied_current_files':current,'historical_manifest_hashes':{name:{'script_sha256':json.loads((ROOT/name/'manifest.json').read_text()).get('script_sha256'),'module_sha256':json.loads((ROOT/name/'manifest.json').read_text()).get('module_sha256')} for name in names},'warning':'Historical manifests are unchanged. Copied files are the analysis-time harness, not assumed byte-identical to every historical run. Historical script source 0dc6... is not reconstructed.'})

def analyze(names):
    group_rows=[];pair_rows=[];trial_rows=[];background_rows=[];consistency=[];negative_rows=[];pressure_rows=[];derived={}
    for name in names:
        trials=trials_for(name);summary=json.loads((ROOT/name/'summary.json').read_text());manifest=json.loads((ROOT/name/'manifest.json').read_text());groups=collections.defaultdict(list);pairs=collections.defaultdict(dict)
        for t in trials:
            key=json.dumps([t['prefix_tokens'],t['tool_wait_s'],t['lead_s'],t['group']]);groups[key].append(t);pairs[t['pair']][t['group']]=t
            cache=t.get('resume_cache',{});load=t.get('background_load') or {};pressure=t.get('host_pressure') or {}
            trial_rows.append({'run':name,'trial':t['id'],'pair':t['pair'],'prefix_tokens':t['prefix_tokens'],'wait_s':t['tool_wait_s'],'lead_s':t['lead_s'],'group':t['group'],'valid':t.get('valid',False),'invalid_reasons':json.dumps(t.get('invalid_reasons',[]),ensure_ascii=False),'resumed':bool(t.get('resume')),'ttft_ms':t.get('resume',{}).get('ttft_ms'),'cycle_ms':t.get('complete_cycle_ms'),'quiescent_cycle_ms':t.get('quiescent_cycle_ms'),'ready_before_resume':bool(t.get('prefetch_ready_before_resume',t.get('prefetch_ready_by_resume',False))),'device_hit_tokens':cache.get('device'),'cpu_hit_tokens':cache.get('cpu'),'storage_hit_tokens':cache.get('storage'),'repeated_prefill_tokens':t.get('repeated_prefill_tokens'),'background_total_failures':load.get('total_failures'),'background_in_window_failures':load.get('in_window_failures'),'background_drained_after_cutoff':load.get('drained_after_cutoff')})
            if pressure:
                pressure_rows.append({'run':name,'trial':t['id'],'group':t['group'],'pressure_prefixes':len(pressure),'requested_pressure_tokens_sum':sum(x.get('tokens',0) for x in pressure),'per_prefix_observed_host_tokens':[x.get('host_tokens_observed') for x in pressure],'generation_latency_ms_sum':sum(x.get('generation',{}).get('latency_ms',0) for x in pressure),'transfer_elapsed_ms_sum':sum(x.get('transfer',{}).get('elapsed_ms',0) for x in pressure),'note':'each prefix inspected after its own demote; not simultaneous byte-peak telemetry; setup excluded from complete_cycle_ms'})
            if t['group'] in NEG:negative_rows.append({'run':name,'trial':t['id'],'group':t['group'],'scenario_valid':t.get('valid',False),'resumed':bool(t.get('resume')),'invalid_reasons':t.get('invalid_reasons',[]),'note':'valid is scenario validation; no-resume cases do not contribute latency distributions'})
            if load:
                requests=load.get('requests',[])
                background_rows.append({'run':name,'trial':t['id'],'prefix_tokens':t['prefix_tokens'],'group':t['group'],'foreground_valid':t.get('valid',False),'concurrency':load.get('concurrency'),'measurement_s':load.get('measurement_s'),'started_requests':load.get('started_requests'),'in_window_completed':load.get('in_window_completed'),'in_window_successful':load.get('in_window_successful'),'in_window_failures':load.get('in_window_failures'),'total_failures':load.get('total_failures'),'drained_after_cutoff':load.get('drained_after_cutoff'),'drained_failures':load.get('drained_failures'),'requests_per_s':load.get('completion_cohort_requests_per_s'),'background_ttft_p50_ms':quantile([r['ttft_ms'] for r in requests if r.get('success') and 'ttft_ms' in r],.5),'background_latency_p50_ms':quantile([r['latency_ms'] for r in requests if r.get('success') and 'latency_ms' in r],.5)})
        for label,ts in sorted(groups.items()):
            gs=summary['groups'][label];resumed=[t for t in ts if t.get('resume')];values=[t['resume']['ttft_ms'] for t in resumed]
            check=(len(ts)==gs['attempts'] and len(resumed)==gs['completed_resumes'] and (not values or math.isclose(statistics.mean(values),gs['resumed_ttft_ms']['mean'],abs_tol=1e-7)))
            consistency.append({'run':name,'group':label,'raw_summary_match':check})
            if not check:raise RuntimeError(f'summary mismatch {name} {label}')
            tokens,wait,lead,group=json.loads(label)
            group_rows.append({'run':name,'runtime_commit':OLD if name.startswith('bench-p2-') else NEW,'prefix_tokens':tokens,'wait_s':wait,'lead_s':lead,'group':group,'attempts':len(ts),'scenario_valid':sum(t.get('valid',False) for t in ts),'completed_resumes':len(resumed),'validated_resumes':gs['valid'],'ttft_mean_ms':gs['resumed_ttft_ms']['mean'],'ttft_p50_ms':gs['resumed_ttft_ms']['p50'],'ttft_p95_ms':gs['resumed_ttft_ms']['p95'],'ttft_p99_ms':gs['resumed_ttft_ms']['p99'],'cycle_mean_ms':gs['complete_cycle_ms']['mean'],'cycle_p50_ms':gs['complete_cycle_ms']['p50'],'ready_by_resume':gs.get('prefetch_ready_by_resume',0),'cpu_hit_tokens_sum':gs.get('tier_hit_tokens',{}).get('cpu'),'storage_hit_tokens_sum':gs.get('tier_hit_tokens',{}).get('storage'),'background_failures':gs.get('background_failures',0)})
        comparisons=[('on_demand',g) for g in manifest['groups'] if g!='on_demand']
        if 'pin_demote' in manifest['groups']:comparisons += [('demote','pin_demote'),('pin_demote','pin_demote_prefetch')]
        blocks={}
        for base,candidate in comparisons:
            by_label=collections.defaultdict(list)
            for pair,members in pairs.items():
                a,b=members.get(base),members.get(candidate)
                if a and b and a.get('resume') and b.get('resume'):
                    by_label[json.dumps([b['prefix_tokens'],b['tool_wait_s'],b['lead_s'],candidate])].append((a,b))
            for label,ab in by_label.items():
                dt=[b['resume']['ttft_ms']-a['resume']['ttft_ms'] for a,b in ab];dc=[b['complete_cycle_ms']-a['complete_cycle_ms'] for a,b in ab]
                original=summary['paired'].get(label) if base=='on_demand' else None
                boot_t=original.get('paired_mean_ttft_bootstrap') if original else None;boot_c=original.get('paired_mean_cycle_bootstrap') if original else None
                t_ci=boot_t or bootstrap(dt,label if base=='on_demand' else label+base);c_ci=boot_c or bootstrap(dc,label+'cycle' if base=='on_demand' else label+base+'cycle')
                token,wait,lead,_=json.loads(label);r={'run':name,'prefix_tokens':token,'wait_s':wait,'lead_s':lead,'candidate':candidate,'baseline':base,'pairs':len(ab),'valid_pairs':sum(a.get('valid',False) and b.get('valid',False) for a,b in ab),'ttft_delta_mean_ms':statistics.mean(dt),'ttft_ci_low_ms':t_ci['ci'][0],'ttft_ci_high_ms':t_ci['ci'][1],'cycle_delta_mean_ms':statistics.mean(dc),'cycle_ci_low_ms':c_ci['ci'][0],'cycle_ci_high_ms':c_ci['ci'][1],'ttft_ci_origin':'original_summary' if boot_t else 'offline_recomputed_from_raw','cycle_ci_origin':'original_summary' if boot_c else 'offline_recomputed_from_raw','bootstrap_resamples':4000,'ttft_bootstrap_seed':t_ci['seed'],'cycle_bootstrap_seed':c_ci['seed']}
                pair_rows.append(r);blocks[label+' vs '+base]={'row':r,'ttft_paired_deltas_ms':dt,'cycle_paired_deltas_ms':dc}
        derived[name]=blocks
    write_csv(ROOT/'groups.csv',group_rows);write_csv(ROOT/'paired.csv',pair_rows);write_csv(ROOT/'trials.csv',trial_rows);write_csv(ROOT/'background.csv',background_rows)
    dump(ROOT/'paired-data.json',derived);dump(ROOT/'negative-scenarios.json',negative_rows);dump(ROOT/'host-pressure.json',pressure_rows);dump(ROOT/'validation.json',{'raw_to_summary_checks':consistency,'all_checks_passed':all(c['raw_summary_match'] for c in consistency),'runs':names,'warning':'summary groups.valid counts valid resumed requests; scenario-valid negative trials may have zero resumed requests'})
    if background_rows:
        agg=[]
        for name in names:
            for token in [8192,16384]:
                for group in ['on_demand','prefetch']:
                    rs=[r for r in background_rows if r['run']==name and r['prefix_tokens']==token and r['group']==group]
                    if rs:agg.append({'run':name,'prefix_tokens':token,'group':group,'foreground_trials':len(rs),'measurement_seconds_sum':sum(r['measurement_s'] for r in rs),'in_window_successful_sum':sum(r['in_window_successful'] for r in rs),'duration_weighted_requests_per_s':sum(r['in_window_successful'] for r in rs)/sum(r['measurement_s'] for r in rs),'mean_trial_requests_per_s':statistics.mean(r['requests_per_s'] for r in rs),'total_background_failures':sum(r['total_failures'] for r in rs),'in_window_failures':sum(r['in_window_failures'] for r in rs),'drained_after_cutoff':sum(r['drained_after_cutoff'] for r in rs),'drained_failures':sum(r['drained_failures'] for r in rs)})
        dump(ROOT/'background-aggregate.json',agg)
    return group_rows,pair_rows

def plots(groups,pairs):
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':170})
    fig,axs=plt.subplots(1,2,figsize=(10,4.0),layout='constrained')
    for idx,metric in enumerate(['ttft','cycle']):
        ax=axs[idx]
        for off,name,color,label in [(-.10,'bench-p2-matrix-v1','#a74338','P3 snapshot 69de (N=6)'),(.10,'bench-optimized-matrix-v1','#24698c','Snapshot 630da (N=10)')]:
            rs=sorted((r for r in pairs if r['run']==name and r['baseline']=='on_demand' and r['candidate']=='prefetch'),key=lambda r:r['prefix_tokens']);x=[i+off for i in range(len(rs))];y=[r[metric+'_delta_mean_ms'] for r in rs];error=[[r[metric+'_delta_mean_ms']-r[metric+'_ci_low_ms'] for r in rs],[r[metric+'_ci_high_ms']-r[metric+'_delta_mean_ms'] for r in rs]]
            ax.errorbar(x,y,yerr=error,fmt='o',capsize=4,label=label,color=color)
        ax.axhline(0,color='#666666',lw=.8);ax.set_xticks([0,1,2],['2K','8K','16K']);ax.set_xlabel('Published prefix');ax.set_ylabel('Paired mean delta (ms)');ax.set_title('Resumed TTFT' if metric=='ttft' else 'Complete foreground cycle');ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('Same 1 s wait / 0.5 s lead: prefetch minus on-demand\n95% paired bootstrap intervals; cross-run code and harness differ',fontsize=10)
    fig.savefig(ROOT/'short-window.png',metadata={'Software':'Offline benchmark artifact analysis'});plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');groupnames=['on_demand','prefetch','demote','pin_demote','pin_demote_prefetch'];colors=['#697783','#207f9e','#c88944','#a45d68','#5c9171']
    for j,(g,c) in enumerate(zip(groupnames,colors)):
        rs=sorted((r for r in groups if r['run']=='bench-optimized-retention-matrix' and r['group']==g),key=lambda r:r['prefix_tokens'])
        for ax,metric in zip(axs,['ttft_mean_ms','cycle_mean_ms']):ax.bar([i+(j-2)*.15 for i in range(3)],[r[metric] for r in rs],width=.14,color=c,label=g)
    for ax,title in zip(axs,['Resumed TTFT','Complete foreground cycle']):ax.set_xticks([0,1,2],['2K','8K','16K']);ax.set_title(title);ax.set_ylabel('Mean latency (ms)');ax.grid(axis='y',alpha=.15)
    axs[1].legend(fontsize=7);fig.suptitle('Snapshot 630da, 3 s wait / 2.5 s lead, N=4 per cell\nNo host pressure; these runs do not establish PIN retention benefit',fontsize=10);fig.savefig(ROOT/'five-groups.png',metadata={'Software':'Offline benchmark artifact analysis'});plt.close(fig)

def checksums():
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    (ROOT/'SHA256SUMS').write_text(''.join(f'{sha(p.read_bytes())}  {p.relative_to(ROOT).as_posix()}\n' for p in paths))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repackage',action='store_true');parser.add_argument('--include-host-pressure',action='store_true');args=parser.parse_args();names=list(RUNS)
    if args.include_host_pressure or (ROOT/OPTIONAL/'summary.json').exists():names.append(OPTIONAL)
    if args.repackage:package(names)
    groups,pairs=analyze(names);plots(groups,pairs);checksums();print(json.dumps({'runs':len(names),'groups':len(groups),'comparisons':len(pairs),'output':str(ROOT)}))
if __name__=='__main__':main()

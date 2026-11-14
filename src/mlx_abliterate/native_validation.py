"""Isolated native-runner parity, quality and speed validation.

Does not launch servers, import models, or modify installed tags. Inference
entrypoints require an already-running isolated server and explicit tags.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .benchmark import _json_request, run_benchmark, compare_reports, load_prompts
from .evaluate import load_dataset_splits
from .study_evaluation import load_rubrics, score_response
from .study import passes_retention


def validate_isolation(original_url, isolated_url, models_dir=None, original_models_dir=None):
    ports=[]
    for url in (original_url,isolated_url):
        parsed=urlsplit(url)
        if (parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1')
                or parsed.username or parsed.password or parsed.path not in ('','/')
                or parsed.query or parsed.fragment or parsed.port is None):
            raise ValueError('Require explicit loopback HTTP origins with ports')
        ports.append(parsed.port)
    if ports[0]==ports[1]:raise ValueError('Original and isolated server ports must differ')
    if models_dir is not None:
        target=Path(models_dir).resolve()
        original=Path(original_models_dir or Path.home()/'.ollama/models').resolve()
        if target==original or target.is_relative_to(original) or original.is_relative_to(target):
            raise ValueError('Isolated models store must be separate from original store')
    return {'original_port':ports[0],'isolated_port':ports[1],'stores_separate':True if models_dir is not None else None}


def compare_exact_responses(baseline,candidate):
    reasons=[]
    for key in ('settings','prompt_sha256'):
        if key not in baseline or baseline.get(key)!=candidate.get(key):reasons.append('mismatched '+key)
    if baseline.get('status')!='complete' or candidate.get('status')!='complete':reasons.append('incomplete benchmark')
    tested=0
    for phase in ('cold_samples','warmup_samples','samples'):
        left,right=baseline.get(phase,[]),candidate.get(phase,[])
        if len(left)!=len(right) or not left:reasons.append('missing or mismatched '+phase)
        for a,b in zip(left,right):
            tested+=1
            if a.get('status')!='completed' or b.get('status')!='completed':reasons.append('generation failure')
            for key in ('response_sha256','thinking_sha256','eval_count','prompt_eval_count','prompt_index','phase','repetition'):
                if key not in a or a.get(key)!=b.get(key):reasons.append('mismatched '+key)
    return {'passed':not reasons,'paired_samples':tested,'reasons':sorted(set(reasons)),
            'scope':'Exact response hashes and token counts for this fixed short prompt schedule only.'}


def compare_native_speed(baseline,candidate):
    result=compare_reports(baseline,candidate)
    # Patched versus stock runner version is intentional. Preserve both version
    # strings; every other matcher from the existing speed gate stays mandatory.
    versions={'original':baseline.get('ollama_version'),'isolated':candidate.get('ollama_version')}
    if all(isinstance(v,str) and v for v in versions.values()):
        reasons=[r for r in result['reasons'] if r!='mismatched or missing ollama_version']
        ratio=result['decode_ratio']
        result.update(reasons=reasons,status='inconclusive' if reasons else ('pass' if ratio is not None and ratio>=.9 else 'fail'))
    result['runtime_versions']=versions
    result['intentional_runtime_comparison']=True
    return result


def quality_gate(baseline,edited,expected_ids):
    for rows in (baseline,edited):
        if len(rows)!=len(expected_ids) or {r['id'] for r in rows}!=set(expected_ids):
            raise ValueError('Served task coverage mismatch')
    originals={r['id']:r for r in baseline}
    current={r['id']:r for r in edited}
    regressed=sorted(k for k,v in originals.items() if v.get('task_check_passed') and
                     (not current[k].get('task_check_passed') or current[k]['status']!='completed'))
    completed=all(r['status']=='completed' for r in baseline+edited)
    aligned=[current[r['id']] for r in baseline]
    retained=passes_retention(baseline,aligned)
    return {'passed':completed and retained,'all_tasks_completed':completed,
            'baseline_successes_retained':retained,'regressed_task_ids':regressed,
            'stock_correct':sum(bool(r.get('task_check_passed')) for r in baseline),
            'edited_correct':sum(bool(r.get('task_check_passed')) for r in edited),
            'tasks':len(expected_ids)}


class Servers:
    def __init__(self,original_url,isolated_url):
        self.original=original_url.rstrip('/');self.isolated=isolated_url.rstrip('/')
        self.isolation=validate_isolation(self.original,self.isolated)
    def api(self,url,path,payload=None):
        return _json_request(url+path,payload,180)
    def empty(self):
        if any(self.api(url,'/api/ps')['models'] for url in (self.original,self.isolated)):
            raise ValueError('Unload resident models on both servers before validation')
    def digest(self,url,tag):
        rows=[r for r in self.api(url,'/api/tags')['models'] if tag in (r.get('name'),r.get('model'))]
        if len(rows)!=1 or not rows[0].get('digest'):raise ValueError('Expected exact model tag digest')
        return rows[0]['digest']
    def unload(self,url,tag):
        result=self.api(url,'/api/generate',{'model':tag,'stream':False,'keep_alive':0})
        if result.get('done') is not True:raise ValueError('Model unload failed')


def run_parity(original_url,isolated_url,original_tag,isolated_stock_tag,zero_tag,prompts):
    servers=Servers(original_url,isolated_url);servers.empty()
    reports={};digests={}
    for arm,url,tag in [('original',servers.original,original_tag),('isolated_stock',servers.isolated,isolated_stock_tag),('zero',servers.isolated,zero_tag)]:
        servers.empty();digests[arm]=servers.digest(url,tag)
        try:
            reports[arm]=run_benchmark(tag,prompts,base_url=url,repeats=1,warmups=1,cold=True,
                options={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':256})
        finally:servers.unload(url,tag)
        if servers.digest(url,tag)!=digests[arm]:raise ValueError('Tag changed during parity validation')
    checks={arm:compare_exact_responses(reports['original'],reports[arm]) for arm in ('isolated_stock','zero')}
    return {'schema_version':1,'passed':all(r['passed'] for r in checks.values()),
            'digests':digests,'checks':checks,'reports':reports,'isolation':servers.isolation}


def run_quality(original_url,isolated_url,original_tag,edited_tag,dataset,rubric_file):
    servers=Servers(original_url,isolated_url);servers.empty()
    data=load_dataset_splits(dataset);rubrics=load_rubrics(Path(rubric_file),data)
    tasks=[r for r in list(data.splits['final'])+list(data.splits['capability'])
           if rubrics['rubrics'][r['id']]['kind']!='refusal_likelihood']
    if len(tasks)!=12:raise ValueError('Expected twelve served exact-answer tasks')
    reports={}
    for arm,url,tag in [('stock',servers.original,original_tag),('edited',servers.isolated,edited_tag)]:
        servers.empty();digest=servers.digest(url,tag);results=[]
        try:
            for row in tasks:
                rubric=rubrics['rubrics'][row['id']]
                try:
                    response=servers.api(url,'/api/chat',{'model':tag,'messages':[{'role':'user','content':row['prompt']}],
                        'think':False,'stream':False,'options':{'temperature':0,'seed':42,'num_ctx':4096,'num_predict':96}})
                    if response.get('done') is not True or response.get('done_reason') not in ('stop','length'):
                        raise ValueError('Unknown completion')
                    status='completed' if response['done_reason']=='stop' else 'truncated'
                    scored=score_response({'status':status,'text':response.get('message',{}).get('content','')},rubric)
                except Exception as error:
                    scored=score_response({'status':'runtime_error'},rubric);scored['error_type']=type(error).__name__
                results.append({'id':row['id'],'kind':rubric['kind'],**scored})
        finally:servers.unload(url,tag)
        if servers.digest(url,tag)!=digest:raise ValueError('Tag changed during quality validation')
        reports[arm]={'tag':tag,'model_digest':digest,'results':results,
                      'correct':sum(bool(r.get('task_check_passed')) for r in results),'tasks':len(results)}
    gate=quality_gate(reports['stock']['results'],reports['edited']['results'],{r['id'] for r in tasks})
    return {'schema_version':1,'dataset_fingerprint':data.fingerprint,'rubric_sha256':rubrics['sha256'],
            'models':reports,'isolation':servers.isolation,**gate,
            'scope':'Twelve served authored exact-answer checks; no broad capability claim.'}


def run_speed(original_url,isolated_url,original_tag,edited_tag,prompts,*,blocks=1):
    if type(blocks) is not int or blocks not in (1,2,4):
        raise ValueError('Speed blocks must be 1, 2 or 4; multiple blocks balance model order')
    import copy
    from .benchmark import _summary
    servers=Servers(original_url,isolated_url);servers.empty()
    reports={};digests={};block_reports=[]
    arms=[('stock',servers.original,original_tag),('edited',servers.isolated,edited_tag)]
    for block in range(blocks):
        measured={}
        order=arms if block%2==0 else list(reversed(arms))
        for arm,url,tag in order:
            servers.empty();digest=servers.digest(url,tag)
            if arm in digests and digests[arm]!=digest:raise ValueError('Tag changed between speed blocks')
            digests[arm]=digest
            try:
                measured[arm]=run_benchmark(tag,prompts,base_url=url,repeats=5,warmups=1,cold=True,
                    options={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':256})
            finally:servers.unload(url,tag)
            if servers.digest(url,tag)!=digest:raise ValueError('Tag changed during speed validation')
            raw=measured[arm]
            if arm not in reports:
                reports[arm]=copy.deepcopy(raw)
                for phase in ('samples','cold_samples','warmup_samples'):reports[arm][phase]=[]
            combined=reports[arm]
            for key in ('settings','prompt_sha256','environment','ollama_version'):
                if combined[key]!=raw[key]:raise ValueError('Speed conditions changed between blocks')
            for phase in ('samples','cold_samples','warmup_samples'):
                combined[phase].extend({**row,'block':block,'repetition':block*5+row['repetition']} for row in raw[phase])
        block_reports.append({'block':block,'order':[a[0] for a in order],
            'comparison':compare_native_speed(measured['stock'],measured['edited']),
            'reports':measured})
    for arm,combined in reports.items():
        combined['settings']={**combined['settings'],'repeats':5*blocks,'blocks':blocks,
            'warmups_per_block':1,'cold_per_block':True}
        combined['summary']=_summary(combined['samples'])
        combined['status']='complete' if all(b['reports'][arm]['status']=='complete' for b in block_reports) else 'incomplete'
        combined['serving_configuration_sha256']=hashlib.sha256(json.dumps({key:combined[key] for key in
            ('ollama_version','settings','prompt_sha256','environment')},sort_keys=True,allow_nan=False).encode()).hexdigest()
    comparison=compare_native_speed(reports['stock'],reports['edited'])
    if any(report['status']!='complete' for report in reports.values()):
        comparison['reasons']=sorted(set(comparison['reasons']+['incomplete cold, warm-up or measured generation']))
        comparison['status']='inconclusive'
    return {'schema_version':1,'passed':comparison['status']=='pass','comparison':comparison,
            'reports':reports,'digests':digests,'isolation':servers.isolation,
            'blocks':block_reports,'scope':'All measured warm samples included; alternating arm order when blocks > 1.'}



def verify_native_import(model, export_directory, source_export, models_dir, *, original_models_dir=None):
    """CPU-only full imported-payload check in a separate explicit model store."""
    from .native_edit_export import verify_native_export
    from .export import verify_import
    from .inventory import inspect_model
    validate_isolation('http://127.0.0.1:11434','http://127.0.0.1:11435',models_dir,original_models_dir)
    native=verify_native_export(Path(export_directory),Path(source_export))
    before=inspect_model(model,Path(models_dir))['manifest_digest']
    imported=verify_import(model,Path(export_directory),Path(models_dir))
    after=inspect_model(model,Path(models_dir))['manifest_digest']
    if before!=after:raise ValueError('Imported manifest changed during verification')
    return {'schema_version':1,'passed':True,'model':model,'imported_manifest_digest':after,
            'native_artifact':native,'import_integrity':imported,
            'scope':'CPU payload/config identity only; generation parity, quality and speed are separate gates.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['parity','quality','speed','verify-import'])
    parser.add_argument('--original-url',default='http://127.0.0.1:11434')
    parser.add_argument('--isolated-url',default='http://127.0.0.1:11435')
    parser.add_argument('--original-tag',default='qwen3.8:27b-mlx')
    parser.add_argument('--isolated-stock-tag',default='native-stock:27b')
    parser.add_argument('--zero-tag',default='native-zero:27b')
    parser.add_argument('--edited-tag',default='native-edited:27b')
    parser.add_argument('--models-dir',type=Path)
    parser.add_argument('--native-export',type=Path)
    parser.add_argument('--source-export',type=Path)
    parser.add_argument('--prompts',type=Path)
    parser.add_argument('--speed-blocks',type=int,choices=[1,2,4],default=1)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--rubrics',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():raise FileExistsError(args.output)
    if args.stage=='verify-import':
        if args.models_dir is None or args.native_export is None or args.source_export is None:
            parser.error('verify-import requires --models-dir, --native-export and --source-export')
        report=verify_native_import(args.edited_tag,args.native_export,args.source_export,args.models_dir)
    elif args.stage=='quality':
        if args.dataset is None or args.rubrics is None:parser.error('quality requires --dataset and --rubrics')
        report=run_quality(args.original_url,args.isolated_url,args.original_tag,args.edited_tag,args.dataset,args.rubrics)
    else:
        if args.prompts is None:parser.error('parity/speed require --prompts')
        prompts=load_prompts(args.prompts)
        if args.stage=='parity':report=run_parity(args.original_url,args.isolated_url,args.original_tag,args.isolated_stock_tag,args.zero_tag,prompts)
        else:report=run_speed(args.original_url,args.isolated_url,args.original_tag,args.edited_tag,prompts,blocks=args.speed_blocks)
    from .cli import write_report
    write_report(report,args.output)
    print(json.dumps({'stage':args.stage,'passed':report['passed']}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())

"""Reviewable isolated native runner validation; never starts a server.

Default prints commands only. Lead must separately start the patched binary with
this isolated store/port and ensure all other model workers are unloaded.
"""
import argparse,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mlx_abliterate.native_validation import validate_isolation,Servers
from mlx_abliterate.cli import write_report
from mlx_abliterate.inventory import inspect_model

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--models-dir',type=Path,required=True)
p.add_argument('--label',default='native-v6')
p.add_argument('--source-export',type=Path,default=ROOT/'outputs/stock-noop')
p.add_argument('--zero-export',type=Path,required=True)
p.add_argument('--edited-export',type=Path,required=True)
p.add_argument('--original-url',default='http://127.0.0.1:11434')
p.add_argument('--isolated-url',default='http://127.0.0.1:11435')
p.add_argument('--stage',choices=['import','verify','parity','standard','quality','speed','all'],default='all')
p.add_argument('--execute',action='store_true')
a=p.parse_args()
if not a.label or not all(c.isalnum() or c in '-_' for c in a.label):raise ValueError('Unsafe label')
validate_isolation(a.original_url,a.isolated_url,a.models_dir)
PYTHON=sys.executable
run=ROOT/'runs'
tags={name:f'{a.label}-{name}:27b' for name in ('stock','zero','edited')}
exports={'stock':a.source_export,'zero':a.zero_export,'edited':a.edited_export}
def result(name):return run/f'{a.label}-{name}.json'
def module(name,*args):return [PYTHON,'-m',name,*map(str,args)]
common=['--original-url',a.original_url,'--isolated-url',a.isolated_url]
commands={
 'import':[[str(a.binary),'create',tags[arm],'-f',str(exports[arm]/'Modelfile')] for arm in ('stock','zero','edited')],
 'verify':[module('mlx_abliterate','verify-import',tags['stock'],a.source_export,'--models-dir',a.models_dir,'--output',result('stock-import'))]+[
   module('mlx_abliterate.native_validation','verify-import','--edited-tag',tags[arm],'--native-export',exports[arm],
     '--source-export',a.source_export,'--models-dir',a.models_dir,'--output',result(arm+'-import')) for arm in ('zero','edited')],
 'parity':[module('mlx_abliterate.native_validation','parity',*common,'--isolated-stock-tag',tags['stock'],
   '--zero-tag',tags['zero'],'--prompts',run/'prompts.json','--output',result('parity'))],
 'standard':[module('mlx_abliterate.heretic_benchmark',tags['edited'],ROOT/'datasets/heretic-standard/harmful-test100.json',
   ROOT/'datasets/heretic-standard/benchmark-config.json',run/(a.label+'-standard100'),
   '--base-url',a.isolated_url,'--output',result('standard100'))],
 'quality':[module('mlx_abliterate.native_validation','quality',*common,'--edited-tag',tags['edited'],
   '--dataset',ROOT/'datasets/benign-sensitive-v1.jsonl','--rubrics',ROOT/'datasets/benign-sensitive-v1.rubrics.json',
   '--output',result('quality'))],
 'speed':[module('mlx_abliterate.native_validation','speed',*common,'--edited-tag',tags['edited'],
   '--prompts',run/'prompts.json','--output',result('speed'))]}
stages=list(commands) if a.stage=='all' else [a.stage]
for stage in stages:
 print('STAGE',stage)
 for command in commands[stage]:print(shlex.join(command))
if not a.execute:raise SystemExit(0)
if not a.binary.is_file():raise ValueError('Patched binary missing')
servers=Servers(a.original_url,a.isolated_url)
servers.empty()
original_digest=servers.digest(a.original_url,'qwen3.8:27b-mlx')
original_manifest=inspect_model('qwen3.8:27b-mlx')['manifest_digest']
environment=os.environ.copy();environment['OLLAMA_HOST']=a.isolated_url;environment['OLLAMA_MODELS']=str(a.models_dir.resolve())
manifest={'schema_version':1,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
          'isolation':validate_isolation(a.original_url,a.isolated_url,a.models_dir),
          'original_model_digest':original_digest,'original_manifest_digest':original_manifest,
          'tags':tags,'source_exports':{arm:hashlib.sha256((path/'config.json').read_bytes()).hexdigest() for arm,path in exports.items()}}
admission=result('admission')
if admission.exists():
 if json.loads(admission.read_text())!=manifest:raise ValueError('Native pipeline admission changed')
else:write_report(manifest,admission)
for stage in stages:
 if stage=='import':
  existing=servers.api(servers.isolated,'/api/tags')['models']
  if any(tag in (row.get('name'),row.get('model')) for tag in tags.values() for row in existing):raise ValueError('Isolated target tags already exist; use verify stage, never overwrite')
 if stage=='parity':
  for arm in ('stock','zero','edited'):
   evidence=json.loads(result(arm+'-import').read_text())
   if arm!='stock' and evidence.get('passed') is not True:raise ValueError('Native import gate failed')
   if arm=='stock' and not all(evidence.get(k) is True for k in ('payloads_verified','assets_verified','runtime_metadata_verified')):raise ValueError('Isolated stock import failed')
 if stage in ('standard','quality','speed'):
  if json.loads(result('parity').read_text()).get('passed') is not True:raise ValueError('Stock/zero parity gate required')
 if stage in ('quality','speed'):
  standard=json.loads(result('standard100').read_text())
  if standard['summary']['keyword_rate_target_passed'] is not True or standard['model_digest']!=servers.digest(servers.isolated,tags['edited']):raise ValueError('Current native standard100 gate required')
 if stage=='speed':
  quality=json.loads(result('quality').read_text())
  if quality['passed'] is not True or quality['models']['edited']['model_digest']!=servers.digest(servers.isolated,tags['edited']):raise ValueError('Current native served quality gate required')
 try:
  for command in commands[stage]:subprocess.run(command,cwd=ROOT,env=environment,check=True)
 finally:
  if stage=='standard':servers.unload(servers.isolated,tags['edited'])
 if servers.digest(servers.original,'qwen3.8:27b-mlx')!=original_digest or inspect_model('qwen3.8:27b-mlx')['manifest_digest']!=original_manifest:
  raise ValueError('Original stock identity changed')
 if stage=='verify':
  for arm in ('stock','zero','edited'):
   # Bind HTTP digest to the manifest just verified in the isolated disk store.
   disk=inspect_model(tags[arm],a.models_dir)['manifest_digest'].removeprefix('sha256:')
   served=servers.digest(servers.isolated,tags[arm]).removeprefix('sha256:')
   if disk!=served:raise ValueError('Isolated server is not serving verified model store')
print('Requested isolated validation stages finished; no original tags or server configuration modified.')

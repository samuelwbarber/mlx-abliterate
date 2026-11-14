import json,os,signal,subprocess,sys,time,urllib.request,re
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])
label=sys.argv[1];command=sys.argv[2:]
if not label or not all(c.isalnum() or c in '-_' for c in label):raise SystemExit('Unsafe label')
model_label=os.environ.get('MLX_ABLITERATE_MODEL_LABEL', label)
record=Path('runs')/(label+'-supervision.json');log=Path('runs')/(label+'-pipeline.log')
if record.exists() or log.exists():raise SystemExit('Existing native supervision artifact')
roots=[]
for port in (11434,11435):
 ids=subprocess.check_output(['lsof','-t',f'-iTCP:{port}','-sTCP:LISTEN'],text=True).split()
 if len(ids)!=1:raise SystemExit('Expected one listener per server')
 roots.append(int(ids[0]))
def snapshot():
 rows={}
 for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines():
  pid,parent,rss=map(int,line.split());rows[pid]=(parent,rss*1024)
 owned=set(roots)
 while True:
  more={pid for pid,(parent,_) in rows.items() if parent in owned}
  if more.issubset(owned):break
  owned|=more
 return {str(pid):rows[pid][1] for pid in owned if pid in rows}
def swaps():
 text=subprocess.check_output(['vm_stat'],text=True)
 return int(re.search(r'^Swapouts:\s+(\d+)',text,re.M).group(1))
def unload():
 for port,tags in ((11434,['qwen3.8:27b-mlx']),(11435,[f'{model_label}-{x}:27b' for x in ('stock','zero','edited')])):
  for tag in tags:
   try:
    req=urllib.request.Request(f'http://127.0.0.1:{port}/api/generate',data=json.dumps({'model':tag,'keep_alive':0}).encode(),headers={'Content-Type':'application/json'})
    urllib.request.urlopen(req,timeout=10).read()
   except Exception:pass
samples=[];failure=None;start=time.monotonic();initial_swaps=swaps()
with log.open('x') as output:
 p=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while p.poll() is None:
   processes=snapshot();pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
   samples.append({'seconds':time.monotonic()-start,'server_tree_rss_bytes':sum(processes.values()),'process_rss_bytes':processes,'pressure':pressure})
   if samples[-1]['server_tree_rss_bytes']>34*1024**3 or pressure!=1 or time.monotonic()-start>3600:
    failure='server_tree_memory_pressure_or_timeout';break
   time.sleep(1)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  if failure or p.returncode:unload()
record.write_text(json.dumps({'schema_version':1,'exitcode':p.returncode,'failure':failure,'server_pids':roots,'samples':samples,'initial_system_swapouts':initial_swaps,'final_system_swapouts':swaps(),'scope':'Observed RSS of both server process trees, pressure and global swapout counters; RSS is not a direct Metal allocation measurement.'},indent=2))
print('Native pipeline exit',p.returncode,'failure',failure,'seconds',time.monotonic()-start)
raise SystemExit(p.returncode or (1 if failure else 0))

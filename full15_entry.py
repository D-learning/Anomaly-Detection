"""Persist actual model exit status even when the dispatcher disconnects."""
import json,pathlib,subprocess,sys,datetime,traceback,os
request=pathlib.Path(sys.argv[1]);job=json.loads(request.read_text());root=request.parent
def atomic(name,obj):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
    p=subprocess.Popen(job['command'],cwd=job['cwd'],stdin=subprocess.DEVNULL)
    atomic('process.json',dict(model_pid=p.pid,entry_pid=os.getpid(),started=start))
    code=p.wait()
except Exception:
    traceback.print_exc();code=127
atomic('exit.json',dict(exit_code=code,started=start,ended=datetime.datetime.now(datetime.timezone.utc).isoformat()))
raise SystemExit(code)

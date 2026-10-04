import sys,platform,json,subprocess,datetime
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'logs/environment_current.json'
r={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform()}
for name,cmd in [('pip_freeze',[sys.executable,'-m','pip','freeze']),('nvidia_smi',['nvidia-smi'])]:
    try:r[name]=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT)
    except Exception as e:r[name]=str(e)
p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(p)

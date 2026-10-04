import csv,json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((P/'MANIFEST.json').read_text());bad=[]
 for r in manifest['files']:
  p=P/r['path']
  if not p.is_file() or sha(p)!=r['sha256']:bad.append(r['path'])
 if bad:raise SystemExit('Missing/modified files: '+str(bad))
 print('Package checksums verified:',len(manifest['files']))
 path=P/'results.csv'
 if path.exists():
  rows=list(csv.DictReader(path.open(encoding='utf-8')))
  required={'candidate_id','track','sequence','predicted_activity','model_version','run_version','remarks'}
  assert rows and required<=set(rows[0])
  assert len({r['candidate_id'] for r in rows})==len(rows)
  assert len({r['sequence'] for r in rows})==len(rows)
  assert all(0<=float(r['predicted_activity'])<=1 for r in rows)
  print('Result schema verified:',len(rows),'candidates')
if __name__=='__main__':main()

"""Build audited balanced CSV from lawfully obtained cleaned positive/negative sequence CSVs."""
import argparse,csv,json,random,hashlib
from pathlib import Path
AA=set('ACDEFGHIKLMNPQRSTVWY')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--positive',required=True);ap.add_argument('--negative',required=True);ap.add_argument('--out',default='data/AMP-data.csv');a=ap.parse_args()
 audit={};sets=[]
 for key,path in [('positive',a.positive),('negative',a.negative)]:
  p=Path(path);rows=list(csv.DictReader(p.open(encoding='utf-8-sig')));seqs=[]
  for r in rows:
   s=str(r.get('seq',r.get('sequence',''))).strip().upper()
   if 8<=len(s)<=60 and not set(s)-AA:seqs.append(s)
  unique=set(seqs);audit[key]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw':len(rows),'valid':len(seqs),'unique':len(unique)};sets.append(unique)
 pos,neg=sets;overlap=pos&neg;neg-=pos
 audit['cross_label_overlap_removed']=len(overlap)
 out=[{'seq':s,'AMP':1} for s in sorted(pos)]+[{'seq':s,'AMP':0} for s in sorted(neg)]
 # Historical construction used pandas sample(random_state=42); match its ordering for supplied snapshots.
 import pandas as pd
 df=pd.DataFrame(out).sort_values(['seq','AMP'],ascending=[True,False]).sample(frac=1,random_state=42).reset_index(drop=True)
 p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False)
 audit.update(final_rows=len(df),final_positive=len(pos),final_negative=len(neg),seed=42,output_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 p.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(audit)
if __name__=='__main__':main()

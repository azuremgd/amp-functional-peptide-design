"""Real inference/screening and standardized UTF-8 candidate output (track 1)."""
import argparse, csv, json, random, hashlib, time, sys
from pathlib import Path
from bootstrap import ROOT

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def resolve(path):
    p=Path(path);return p if p.is_absolute() else ROOT/p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['rescore','generate','format-only'],default='rescore')
    ap.add_argument('--config',default='configs/demo.json')
    ap.add_argument('--input',default='results/historical/round5_top200.csv')
    ap.add_argument('--out',default='results.csv')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--n-samples',type=int)
    ap.add_argument('--topk',type=int)
    ap.add_argument('--generator')
    args=ap.parse_args();cfg=json.loads(resolve(args.config).read_text(encoding='utf-8'))
    if args.generator:cfg['generator']=args.generator
    if args.n_samples is not None:cfg['n_samples']=args.n_samples
    if args.topk is not None:cfg['topk']=args.topk
    if cfg['n_samples']<1 or cfg['topk']<1:ap.error('n-samples and topk must be positive')
    start=time.time();seed=int(cfg['seed']);random.seed(seed)
    rows=[];audit={'mode':args.mode,'config':cfg,'python':sys.version,'seed':seed}
    if args.mode=='format-only':
        # Explicit reference conversion; this mode never claims model execution.
        rows=list(csv.DictReader(resolve(args.input).open(encoding='utf-8-sig')))
        audit['inference_executed']=False
    else:
        import numpy as np, torch
        from AMP.models.classifier import AMPActivity
        np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        device='cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu'
        if device=='cpu':torch.set_num_threads(4)
        cpath=resolve(cfg['classifier']);cls=AMPActivity(device=device,model_dir=str(cpath))
        audit.update(device=device,torch=torch.__version__,cuda=torch.version.cuda,inference_executed=True,classifier_sha256=sha(cpath/'model.safetensors'))
        if args.mode=='rescore':
            for i,row in enumerate(csv.DictReader(resolve(args.input).open(encoding='utf-8-sig'))):
                seq=row.get('seq') or row.get('sequence');score=cls.predict_prob(seq)
                rows.append(dict(seq=seq,act=score,reference_act=float(row['act']) if row.get('act') else None))
                if (i+1)%10==0:print(f'Rescored {i+1} sequences',flush=True)
            diffs=[abs(r['act']-r['reference_act']) for r in rows if r['reference_act'] is not None]
            audit['max_absolute_score_difference']=max(diffs,default=0)
            audit['input_sha256']=sha(resolve(args.input))
        else:
            from AMP.tools.export_candidates import load_generator
            from AMP.tools.mine_hard_negatives import generate_seq,sample_constraints_from_stats,max_run_length,max_hydrophobic_run
            gpath=resolve(cfg['generator']);gen,stats=load_generator(str(gpath),device)
            audit['generator_sha256']=sha(gpath);seen=set();counters={'sampled':0,'duplicate':0,'rules_failed':0,'score_failed':0,'passed':0}
            with torch.no_grad():
                for i in range(cfg['n_samples']):
                    counters['sampled']+=1
                    cons=sample_constraints_from_stats(stats,cfg['len_low'],cfg['len_high'])
                    seq=generate_seq(gen,stats,cons,device,cfg['temperature'],cfg['top_p'])
                    if seq in seen:counters['duplicate']+=1;continue
                    seen.add(seq)
                    if not seq or not cfg['len_low']<=len(seq)<=cfg['len_high'] or max_run_length(seq)>cfg['max_run'] or max_hydrophobic_run(seq)>cfg['max_hyd_run']:
                        counters['rules_failed']+=1;continue
                    score=cls.predict_prob(seq)
                    if score<cfg['act_thr']:counters['score_failed']+=1;continue
                    rows.append(dict(seq=seq,act=score,target_properties=cons));counters['passed']+=1
                    if (i+1)%10==0:print(f'Sampled {i+1}/{cfg["n_samples"]}; passed={len(rows)}',flush=True)
            audit['screening_counts']=counters
    if args.mode!='rescore': rows.sort(key=lambda r:float(r['act']),reverse=True)
    if args.mode=='generate':rows=rows[:cfg['topk']]
    if args.mode=='rescore':rows.sort(key=lambda r:float(r['act']),reverse=True)
    seen=set();output=[]
    for r in rows:
        seq=str(r.get('seq') or r.get('sequence')).strip().upper()
        if not seq or set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Invalid amino acid sequence')
        if seq in seen:continue
        seen.add(seq);score=float(r['act'])
        if not 0<=score<=1:raise ValueError('Activity score must be between 0 and 1')
        note={'format-only':'Historical score; no model inference executed','rescore':'Historical candidate re-scored by final classifier','generate':'New small-scale design; not historical Top200'}[args.mode]
        output.append({'candidate_id':f'AMP-{len(output)+1:03d}','track':'赛道一：AI大分子与多肽药物设计','sequence':seq,'predicted_activity':f'{score:.8f}','length_aa':len(seq),'model_version':'generator-round5-tm/classifier-hardneg-r2','run_version':cfg['version'],'remarks':note+'; score is not MIC or experimentally verified activity'})
    dest=resolve(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    fields=['candidate_id','track','sequence','predicted_activity','length_aa','model_version','run_version','remarks']
    with dest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(output)
    audit.update(elapsed_seconds=time.time()-start,output_count=len(output),output_sha256=sha(dest))
    log=ROOT/'logs'/f'{dest.stem}.run.json';log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {len(output)} candidates -> {dest}; run record -> {log}',flush=True)
    if not output:raise SystemExit('No candidate passed the specified rules; do not fabricate or silently relax thresholds.')

if __name__=='__main__':main()

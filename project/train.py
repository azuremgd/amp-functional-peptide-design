"""Seeded training entry for pretraining, supervised training, classifier training or RL."""
import argparse, runpy, sys, os, random, json, time, platform, importlib.metadata
from bootstrap import ROOT

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['pretrain','generator','classifier','optimize'],required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--base-model',default='models/prot_bert_bfd')
    args,other=ap.parse_known_args()
    p=ROOT/args.base_model;os.environ['AMP_BASE_MODEL_DIR']=str(p)
    import numpy as np,torch
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    modules={'pretrain':'pretrain_lm','generator':'train_generator','classifier':'train_classifier','optimize':'train_rl'}
    module='AMP.training.'+modules[args.stage]
    started=time.time()
    record={'stage':args.stage,'seed':args.seed,'arguments':other,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'status':'running'}
    log=ROOT/'logs'/f'new_{args.stage}_{time.time_ns()}.json';log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps(record,indent=2),encoding='utf-8')
    if args.stage in ['pretrain','classifier','generator'] and '--seed' not in other:other+=['--seed',str(args.seed)]
    sys.argv=[module]+other
    # Resolve relative paths consistently from the submission directory.
    os.chdir(ROOT)
    try:
        runpy.run_module(module,run_name='__main__')
    except BaseException as exc:
        record['status']='failed';record['error']=str(exc);raise
    else:record['status']='completed'
    finally:
        record['elapsed_seconds']=time.time()-started
        log.write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':main()

"""Complete small design -> RL optimization -> inference/screening example; no historical claims."""
import argparse,subprocess,sys,os
from bootstrap import ROOT
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3);ap.add_argument('--n-samples',type=int,default=30);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=ap.parse_args()
    env=os.environ.copy()
    if a.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True,env=env)
    # Design with final weights, optimize briefly, then re-generate and screen with updated weights.
    run('design.py','--n-samples','10','--topk','3','--device',a.device,'--out','results/demo_before.csv')
    run('train.py','--stage','optimize','--seed','42','--generator','models/generator_round5.pt','--classifier','models/classifier','--steps',str(a.steps),'--len_low','20','--len_high','30','--out','results/demo_optimized.pt','--log_every','1','--save_every',str(a.steps))
    run('design.py','--generator','results/demo_optimized.pt','--n-samples',str(a.n_samples),'--topk','5','--device',a.device,'--out','results/demo_results.csv')
if __name__=='__main__':main()

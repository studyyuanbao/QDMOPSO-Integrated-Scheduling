"""Explicit training / frozen validation / frozen test command entry points."""
import argparse,random,json,time,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from init3 import Config
from pso_rl_new import run_mopso
import pso_rl_new

def run(mode,checkpoint=None,output='workflow-results',episodes=100,iterations=100,particles=50,seeds=(42,),counts=(1,1,1)):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    # Explicit snapshot of the existing fixed small-factory template.
    source=Path(__file__).with_name('init3.py').read_text(encoding='utf8')
    marker='p1_count, p2_count, p3_count = 1, 1, 1'
    if marker not in source:raise ValueError('Unexpected configuration template')
    namespace={};exec(source.replace(marker,f'p1_count, p2_count, p3_count = {counts[0]}, {counts[1]}, {counts[2]}'),namespace)
    records=[];training=mode=='train';current=checkpoint
    for episode in range(episodes if training else len(seeds)):
        seed=seeds[episode%len(seeds)]+(episode//len(seeds) if training else 0)
        c=namespace['Config']();c.num_particles=particles
        random.seed(seed);np.random.seed(seed)
        before=Path(current).read_bytes() if current else None
        save=folder/f'episode-{episode+1:04d}.json' if training else None
        start=time.perf_counter()
        original_evaluate=pso_rl_new.evaluate
        evaluation_count=[0]
        def counted_evaluate(*args,**kwargs):
            evaluation_count[0]+=1
            return original_evaluate(*args,**kwargs)
        pso_rl_new.evaluate=counted_evaluate
        try:
            result,_,_=run_mopso(c,total_iterations=iterations,load_q_table_path=current,
                                save_q_table_path=str(save) if save else None,exploit_only=not training)
        finally:
            pso_rl_new.evaluate=original_evaluate
        if not training:
            from q_checkpoint import load_checkpoint
            q_before={tuple(r['state']):r['values'] for r in load_checkpoint(current)['q_table']}
            assert dict(result['q_table'])==q_before,'Frozen Q table changed in memory'
            assert Path(current).read_bytes()==before,'Checkpoint file changed'
        records.append({'mode':mode,'episode':episode+1,'seed':seed,'counts':list(counts),'tasks':c.num_tasks,
                        'particles':particles,'iterations':iterations,'nfe':evaluation_count[0],'seconds':time.perf_counter()-start,
                        'q_states':len(result['q_table']),'checkpoint_input':str(current) if current else None,
                        'checkpoint_output':str(save) if save else None,'frozen_unchanged':not training})
        if training:current=str(save)
    (folder/f'{mode}-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    return current,records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','validate','test'])
    p.add_argument('--checkpoint');p.add_argument('--output',default='workflow-results')
    p.add_argument('--episodes',type=int,default=100);p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--particles',type=int,default=50);p.add_argument('--seeds',type=int,nargs='+',default=[42,7,99])
    p.add_argument('--counts',type=int,nargs=3,default=[1,1,1]);a=p.parse_args()
    if a.mode!='train' and not a.checkpoint:p.error('Frozen validation/testing requires --checkpoint')
    if min(a.episodes,a.iterations,a.particles)<=0 or min(a.counts)<0 or sum(a.counts)==0:p.error('Invalid budget/counts')
    run(a.mode,a.checkpoint,a.output,a.episodes,a.iterations,a.particles,a.seeds,a.counts)

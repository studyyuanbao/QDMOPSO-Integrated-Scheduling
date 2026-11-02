from pathlib import Path
import json
class Config:
    def __init__(self):
        d=json.loads((Path(__file__).resolve().parent/'input.json').read_text(encoding='utf-8'))
        self.task_stage=d['task_stage'];self.task_to_product=d['task_to_product']
        for key in ['num_suppliers','precedence','task_to_part','assembly_requirements','product_type_of_instance']:
            setattr(self,key,{int(k):v for k,v in d[key].items()})
        self.machine_task_map={(v['resource'],v['task']):{'time':v['time'],'cost':v['cost']} for v in d['machine_task_map_entries']}
        self.num_tasks=len(self.task_stage);self.num_products=len(self.product_type_of_instance)
        self.num_particles=50;self.num_iterations=100;self.archive_size=400;self.ca_size=200;self.da_size=200
        self.epsilon=.03;self.crossover_prob=.9;self.mutation_prob=.1;self.genetic_interval=10;self.num_genetic_parents=20
        self.resource_min=0;self.resource_max=max(r for rr in self.num_suppliers.values() for r in rr)
        self.product_change_penalty=d['product_change_penalty'];self.variant='full'
        self.task_suppliers={t:self.num_suppliers[t] for t,s in enumerate(self.task_stage) if s=='采购'}
        self.machine_resources={s:sorted({r for t,ss in enumerate(self.task_stage) if ss==s for r in self.num_suppliers[t]}) for s in ['部装','总装']}
        self.global_min_time=self.global_min_cost=float('inf');self.global_max_time=self.global_max_cost=-float('inf')

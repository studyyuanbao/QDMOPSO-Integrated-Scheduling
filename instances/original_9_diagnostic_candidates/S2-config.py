import random


class Config:
    def __init__(self):
        # PSO算法参数
        self.num_particles = 50  # 粒子数量
        self.num_iterations = 100  # 迭代次数
        self.archive_size = 400  # 外部存档大小
        self.epsilon = 0.03  # ε-支配参数
        self.ca_size = 200  # 收敛档案大小
        self.da_size = 200  # 多样性档案大小
        self.crossover_prob = 0.9  # 交叉概率
        self.mutation_prob = 0.1  # 变异概率
        self.genetic_interval = 10  # 遗传操作间隔
        self.num_genetic_parents = 20  # 遗传操作父母数量
        self.resource_min = 0  # 资源最小值
        self.resource_max = 20  # 资源最大值


        # 自适应参数
        self.initial_w = 0.9  # 初始惯性权重
        self.final_w = 0.4  # 最终惯性权重
        self.w_decay_power = 1.5  # 惯性权重衰减指数

        self.initial_c1 = 3 # 初始认知系数
        self.final_c1 = 0.5  # 最终认知系数
        self.c1_decay_power = 1.0  # 认知系数衰减指数

        self.initial_c2 = 0.5  # 初始社会系数
        self.final_c2 = 2.5  # 最终社会系数
        self.c2_growth_power = 1.0  # 社会系数增长指数

        # 突变参数
        self.initial_mutation_rate = 0.4  # 初始突变率
        self.final_mutation_rate = 0.1  # 最终突变率

        # 重启机制
        self.no_improve_threshold = 30  # 无改进迭代次数阈值
        self.reset_ratio = 0.2  # 重启比例

        # 生产调度参数
        self.product_change_penalty = 2  # 产品换型时间惩罚

        # ========= 模板与规模设定 =========
        # 目标规模：P1=3, P2=4, P3=3（原约束与时间/成本模板保持不变）
        p1_count, p2_count, p3_count = 2, 2, 2

        # 每种产品的相对任务阶段模板
        p1_stages = ['采购', '采购', '采购', '部装', '总装']  # 5
        p2_stages = ['采购', '采购', '采购', '采购', '部装', '部装', '总装']  # 7
        p3_stages = ['采购', '采购', '采购', '采购', '采购', '部装', '部装', '部装', '总装']  # 9

        # 相对任务的可用资源（直接复用原始实例的资源配置）
        p1_suppliers = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5, 6], 4: [8]}
        p2_suppliers = {0: [0, 1, 2, 3], 1: [4, 11, 12], 2: [13, 14], 3: [0, 11, 15], 4: [5, 6], 5: [5, 6], 6: [8]}
        p3_suppliers = {0: [0, 1, 2, 3], 1: [4, 11, 12], 2: [13, 14], 3: [0, 11, 15], 4: [0, 3, 4], 5: [5, 6], 6: [5, 6], 7: [5, 6], 8: [8]}

        # 相对任务的加工时间与成本（来自原始单元模板）
        p1_mtm = {(0, 0): {'time': 6.0, 'cost': 10.0}, (1, 0): {'time': 4.0, 'cost': 12.0}, (2, 1): {'time': 5.0, 'cost': 8.0}, (3, 1): {'time': 4.0, 'cost': 11.0}, (4, 2): {'time': 7.0, 'cost': 15.0}, (5, 3): {'time': 7.0, 'cost': 2.0}, (6, 3): {'time': 6.0, 'cost': 3.0}, (8, 4): {'time': 10.0, 'cost': 6.0}}
        p2_mtm = {(0, 0): {'time': 6.0, 'cost': 10.0}, (1, 0): {'time': 5.0, 'cost': 12.0}, (2, 0): {'time': 4.0, 'cost': 9.0}, (3, 0): {'time': 3.0, 'cost': 14.0}, (4, 1): {'time': 6.0, 'cost': 8.0}, (11, 1): {'time': 4.0, 'cost': 11.0}, (12, 1): {'time': 3.0, 'cost': 13.0}, (13, 2): {'time': 5.0, 'cost': 13.0}, (14, 2): {'time': 4.0, 'cost': 15.0}, (0, 3): {'time': 2.0, 'cost': 18.0}, (11, 3): {'time': 4.0, 'cost': 13.0}, (15, 3): {'time': 3.0, 'cost': 16.0}, (5, 4): {'time': 5.0, 'cost': 10.0}, (6, 4): {'time': 6.0, 'cost': 9.0}, (5, 5): {'time': 3.0, 'cost': 5.0}, (6, 5): {'time': 5.0, 'cost': 3.0}, (8, 6): {'time': 8.0, 'cost': 4.0}}
        p3_mtm = {(0, 0): {'time': 6.0, 'cost': 10.0}, (1, 0): {'time': 5.0, 'cost': 12.0}, (2, 0): {'time': 4.0, 'cost': 9.0}, (3, 0): {'time': 3.0, 'cost': 14.0}, (4, 1): {'time': 6.0, 'cost': 8.0}, (11, 1): {'time': 4.0, 'cost': 11.0}, (12, 1): {'time': 3.0, 'cost': 13.0}, (13, 2): {'time': 5.0, 'cost': 12.0}, (14, 2): {'time': 4.0, 'cost': 15.0}, (0, 3): {'time': 2.0, 'cost': 18.0}, (11, 3): {'time': 4.0, 'cost': 13.0}, (15, 3): {'time': 3.0, 'cost': 16.0}, (0, 4): {'time': 3.0, 'cost': 12.0}, (3, 4): {'time': 4.0, 'cost': 13.0}, (4, 4): {'time': 3.0, 'cost': 14.0}, (5, 5): {'time': 3.0, 'cost': 8.0}, (6, 5): {'time': 4.0, 'cost': 6.0}, (5, 6): {'time': 5.0, 'cost': 8.0}, (6, 6): {'time': 6.0, 'cost': 7.0}, (5, 7): {'time': 2.0, 'cost': 7.0}, (6, 7): {'time': 4.0, 'cost': 5.0}, (8, 8): {'time': 8.0, 'cost': 9.0}}

        # 相对任务的前后关系（使用相对索引）
        # 混合前置：保留装配任务内部顺序，不硬性绑定到具体采购任务
        # P1: 部装无硬前置（靠库存控制），总装在部装之后
        p1_prec = {3: [], 4: [3]}
        # P2: 两道部装无硬前置，总装在两道部装之后
        p2_prec = {4: [], 5: [], 6: [4, 5]}
        # P3: 三道部装顺序：第3道在前两道之后；总装在第3道之后
        p3_prec = {5: [], 6: [], 7: [5, 6], 8: [7]}

        # 任务-部件映射（相对任务 -> 部件名）
        p1_parts = {0: 'P1A', 1: 'P1B', 2: 'P1C'}
        p2_parts = {0: 'P2A', 1: 'P2B', 2: 'P2C', 3: 'P2D'}
        p3_parts = {0: 'P3A', 1: 'P3B', 2: 'P3C', 3: 'P3D', 4: 'P3E'}

        # 装配需求（相对任务 -> 需要的部件清单）
        # 产品1：部装需要 P1A、P1B；总装需要 P1C（同时在前置中保留“总装在部装之后”）
        p1_assembly = {3: ['P1A', 'P1B'], 4: ['P1C']}
        p2_assembly = {4: ['P2A', 'P2B'], 5: ['P2C', 'P2D']}
        p3_assembly = {5: ['P3A', 'P3B'], 6: ['P3C', 'P3D'], 8: ['P3E']}

        # ========= 依据规模生成整体实例 =========
        self.task_to_product = []
        self.task_stage = []
        self.num_suppliers = {}
        self.machine_task_map = {}
        self.precedence = {}
        self.task_to_part = {}
        self.assembly_requirements = {}

        # 累积任务偏移量与产品编号
        task_offset = 0
        product_id = 0

        def append_product_instances(count, stages, suppliers, mtm, prec, parts, assembly, part_prefix):
            nonlocal task_offset, product_id
            for _ in range(count):
                # 任务与阶段
                num_local_tasks = len(stages)
                self.task_stage.extend(stages)
                self.task_to_product.extend([product_id] * num_local_tasks)

                # 资源映射与加工参数
                for rel_task, res_list in suppliers.items():
                    self.num_suppliers[task_offset + rel_task] = list(res_list)
                for (res, rel_task), info in mtm.items():
                    self.machine_task_map[(res, task_offset + rel_task)] = {'time': info['time'], 'cost': info['cost']}

                # 前后关系
                for succ_rel, preds_rel in prec.items():
                    self.precedence[task_offset + succ_rel] = [task_offset + p for p in preds_rel]

                # 部件与装配需求
                for rel_task, part_name in parts.items():
                    self.task_to_part[task_offset + rel_task] = part_name
                for rel_task, req_parts in assembly.items():
                    self.assembly_requirements[task_offset + rel_task] = list(req_parts)

                # 偏移与产品编号推进
                task_offset += num_local_tasks
                product_id += 1

        # 依序生成产品1、产品2、产品3的所有实例
        append_product_instances(p1_count, p1_stages, p1_suppliers, p1_mtm, p1_prec, p1_parts, p1_assembly, 'P1')
        append_product_instances(p2_count, p2_stages, p2_suppliers, p2_mtm, p2_prec, p2_parts, p2_assembly, 'P2')
        append_product_instances(p3_count, p3_stages, p3_suppliers, p3_mtm, p3_prec, p3_parts, p3_assembly, 'P3')

        # 汇总数量
        self.num_tasks = len(self.task_stage)
        self.num_products = product_id


        # 初始化全局最优时间和全局最优成本
        self.global_min_time = float('inf')
        self.global_max_time = -float('inf')
        self.global_min_cost = float('inf')
        self.global_max_cost = -float('inf')

        # 随机化部分任务的加工时间
        random.seed(42)  # 确保问题实例生成的一致性
        # for (res, task), info in self.machine_task_map.items():
        #     if self.task_stage[task] == '采购':
        #         base_time = info['time']
        #         factor = random.uniform(0.8, 1.2)
        #         new_time = int(round(base_time * factor))
        #         new_time = max(1, new_time)
        #         self.machine_task_map[(res, task)]['time'] = new_time
        self.enable_time_noise = True  # 控制是否添加扰动
        self.time_noise_range = (0.8, 1.2)  # 控制扰动幅度
        if self.enable_time_noise:
            for (res, task), info in self.machine_task_map.items():
                if self.task_stage[task] == '采购':
                    base_time = info['time']
                    factor = random.uniform(*self.time_noise_range)
                    new_time = base_time * factor  # 直接保留乘法结果（浮点数）
                    new_time = max(1.0, new_time)  # 确保最小值为 1.0（浮点）
                    self.machine_task_map[(res, task)]['time'] = new_time

        # === 为结构学习模块准备 ===
        # 所有采购任务及其资源列表
        self.task_suppliers = {
            t: self.num_suppliers[t]
            for t in range(self.num_tasks)
            if self.task_stage[t] == '采购' and t in self.num_suppliers
        }

        # 不同阶段任务可用机器集合（部装、总装）
        self.machine_resources = {'部装': set(), '总装': set()}
        for t in range(self.num_tasks):
            stage = self.task_stage[t]
            if stage in ['部装', '总装'] and t in self.num_suppliers:
                self.machine_resources[stage].update(self.num_suppliers[t])
        # 转为列表
        self.machine_resources = {
            k: list(v) for k, v in self.machine_resources.items()
        }

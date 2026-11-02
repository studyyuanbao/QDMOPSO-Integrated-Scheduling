# 六实例原参数同预算正式比较

630次运行全部完成：360次Q训练、54次Q测试、216次基线测试，共315万实际目标评价。正式性能样本仅270次测试。原论文参数、修订业务代码，5000评价/次；M1校准、53993测时及旧结果不参与分析。

原始runs/保留每次配置、种子和随机状态、目标与分项费用、完整排程、算子来源、档案事件、逐代状态/动作/奖励/Q更新、TD误差、Q检查点及日志。基线没有Q，不伪造学习数据。Q预训练与测试分开报告，测试探索率0且继续学习，不称冻结策略。

analysis/formal-test-metrics.csv：270次测试指标。case-algorithm-summary.csv：30个实例/算法汇总，标准差是条件测试运行离散程度，不能当成9次独立Q训练。paired-training-group-effects.csv：保留3训练组结构的差值。learning-observations-for-plotting.csv：全部Q逐代观测，回合和训练组明确。

analysis/empirical-reference-*.json：每实例全部正式最终输出合并的经验非支配前沿，固定尺度；不是精确Pareto前沿。metric-protocol.json区分内部HV与分析HV，SP为原始单位欧氏最近邻spacing、不冒称Spread。无效SP留空。

statistics.json：HV为主指标、经验IGD为次指标；每指标4个Q对基线比较单独Holm校正。Friedman仅六个固定实例，渐近p探索性；两两精确符号分配的signed-rank p=0.03125、Holm p=0.125，不声称逐对显著优越，不以不显著推断等效。

record-audit.json记审计范围：全部评价JSON/NFE/分项误差/来源；全部排程gzip CRC和行数、首尾ID；全部学习/档案JSON；Q加载谱系及源码哈希。运行时每次评价已作独立排程约束和费用核对；离线审计不再逐条重新解码。

运行器依赖Python/numpy/scipy/matplotlib等，见environment-analysis.json。原代码可能有额外导入依赖，包内保留冻结源文件，环境不是跨机器自动安装包。路径以控制脚本自身位置推导。重跑会拒绝覆盖runs已存在结果；先阅读protocol.json及controller.py，不要为重跑删除原结果。当前代码和数据仅本地归档，未公开上传。

正式时间包含记录及检查；原100代图表没有被替换或复现。本批新增结果须独立说明，不能把性能差异单因归为预算。详情见既定方案。

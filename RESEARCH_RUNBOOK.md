# Better-1 当前研究运行手册

当前主线已获用户确认：AI风险评分在检查缺失条件下的安全边界与服务率代价。V17影像日期敏感性已完成，832次新拟合，97416行原队列动作与阈值复现V15。运行src/temporal_sensitivity_v17.py（拒绝覆盖）；报告src/report_temporal_v17.py；结果results/temporal_sensitivity_v17；协议reports/v17_temporal_sensitivity_protocol.md。保留外层患者角色，日期排除不等于临床裁定；服务率差G0−G2只在相同队列和评分器内解释，不是相同保证下算法净效应。

V16异常数值缺失化敏感性已完成，624次拟合；结果results/numeric_sensitivity_v16，协议reports/v16_numeric_sensitivity_protocol.md。报告入口src/report_numeric_v16.py；执行入口src/numeric_sensitivity_v16.py拒绝覆盖既有run.json。只改内存副本，CA125原已缺失；实际新增模型输入变化为一例BMI置缺，临床裁定未改变。

V15真实字段删除敏感性已完成，416次新拟合，原配置48708项动作/阈值复现V13。结果results/feature_sensitivity_v15；协议reports/v15_feature_sensitivity_protocol.md。报告入口src/report_feature_sensitivity_v15.py；执行入口src/feature_sensitivity_v15.py拒绝覆盖已有run.json。全部773人保留，删除字段不等于修正来源值或解除临床核实。

V14核实准备包位于results/review_evidence_v14，报告reports/Better1_v14_临床核实分级与敏感性准备.md。执行src/prepare_review_evidence_v14.py仅读取主裁定表与派生数据，输出派生核实副本、重叠计数及假设排除人数，不训练、不改变裁定。主裁定记录仍为results/clinical_review_queue/clinical_review_queue_LOCAL.csv；切勿把副本当成第二个裁定主表。

V13真实全局/子群诊断已完成，复用V6冻结预测，无新增拟合；48708项G2动作核对一致。结果results/real_global_subgroup_v13，协议reports/v13_real_global_subgroup_protocol.md。

```powershell
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\src\report_real_global_v13.py'
```

执行脚本src/real_global_subgroup_v13.py已有run.json时拒绝覆盖；G0证书仅属于总体，子群测试点估计超标不等于真实风险已确认超标。LOCAL动作表不进入对外汇报。最新汇报结论见reports/Better1_导师结论页_V13.md。

V12冻结模型的新种子、多机制复核位于results/frozen_validation_v12/formal，完成状态以该目录run.json为准；协议reports/v12_frozen_validation_protocol.md。

```powershell
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\src\report_frozen_validation_v12.py'
```

运行脚本src/frozen_validation_v12.py直接复用V11模型工厂；已有run.json时拒绝覆盖。报告仅在600组数据、2400次拟合完成后生成。本轮使用新随机种子，但三个生成机制此前均研究过，不能称为未见机制或独立临床验证。

V11可训练评分对照已完成（200次重复、800次拟合），结果位于results/learnable_score_v11/formal；协议reports/v11_learnable_score_protocol.md。报告入口：

```powershell
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\src\report_learnable_score_v11.py'
```

执行脚本src/learnable_score_v11.py支持--smoke，已有run.json时拒绝覆盖。本轮复用V9种子作配对，不能当作新独立验证。所有预处理只在训练角色拟合。

V10可见信息理想分数对照已完成，结果位于results/observed_oracle_v10/formal；协议reports/v10_observed_oracle_protocol.md。生成报告使用：

```powershell
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\src\report_observed_oracle_v10.py'
```

执行脚本src/observed_oracle_v10.py支持--smoke；存在run.json时拒绝覆盖。理想分数依赖已知生成器，仅作模拟诊断；本轮无模型拟合、无新增临床数据。

新增V8/V9使用独立入口，尚未纳入下面的V2–V7统一入口：

```powershell
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\src\audit_service_stability_v8.py'
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\src\report_grid_order_v9.py'
```

V8复核V7的服务稳定性；V9结果已完成，位于results/grid_order_v9/formal。V9执行脚本src/grid_order_v9.py支持--smoke；存在run.json时拒绝覆盖，勿重复启动。V9复用了V7种子来进行配对对照，不算独立验证队列。报告生成不重新训练。

版本化实验入口为 `run_research.py`；无参数仅查看状态，不启动训练。原有 `run_all.py` 是早期 E0/E1/三出路工程入口，其 `all` 不代表本轮 V2–V7 已验证实验。

```powershell
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\run_research.py'
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\run_research.py' --stage v7 --mode run --dry-run
& 'D:\conda\envs\pytorch\python.exe' 'X:\GJA\WORK\run_research.py' --stage v7 --mode report
```

`run` 重跑所选版本并更新其结果；`report` 从已完成结果重新生成报告（V3会重新做短时精确计算，V4无独立报告入口）；`smoke` 仅V5/V7支持，输出各自smoke目录。先用`--dry-run`检查命令。需要保留旧结果时请先归档对应版本目录。

| 版本 | 问题 | 主要输出目录 |
|---|---|---|
| V2 | 数据清洗、缺失增量、评分分布机制、真实认证 | results/selective_risk_v2 |
| V3 | 精确认证功效与合成分数网格边界 | results/boundary_v3 |
| V4 | 训练内预锁定单阈值是否恢复服务 | results/locked_real_v4 |
| V5 | 生成缺失协变量后实际训练模型 | results/mechanism_training_v5/formal |
| V6 | LTT固定序列强基线 | results/fixed_sequence_v6 |
| V7 | 固定模型与顺序，仅改变校准量与信息可得性 | results/calibration_information_v7/formal |

真实数据版本重跑前会比较原始表与派生表的冻结哈希；有变化即停止，需要重新审计。技术审计通过不等于所有临床编码、影像时序和来源疑点已经解决。历史患者级结果文件名含LOCAL，只供本地核查。

入口遇到子程序失败立即停止；状态为RUNNING时阻止重复启动。同一阶段异常退出后若遗留RUNNING，应先查实际进程与错误日志，再决定恢复，不能同时启动第二份结果写入进程。

V5与V7仅模拟数据，不需要GPU。模拟结果不能替代真实外部队列验证，也不能据此直接制定临床补查或招募方案。各版本的研究协议、模型设置和统计限制以reports目录对应protocol和结果报告为准。

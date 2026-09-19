"""Render the executed v2 results without changing experiments or source data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/selective_risk_v2'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)

def table(df):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return 'NA' if pd.isna(x) else f'{x:.4f}'
        return str(x)
    return '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(fmt,row)) + ' |' for row in df.itertuples(index=False,name=None))

d = pd.read_csv(OUT/'e1_paired_deltas.csv')
a = d.groupby(['model','comparison']).agg(AUROC_mean=('delta_AUROC','mean'), AUROC_min=('delta_AUROC','min'), AUROC_max=('delta_AUROC','max'), Brier_mean=('delta_Brier','mean')).reset_index()
b = pd.read_csv(OUT/'e1_association_bootstrap.csv').query("group == 'missing_both'")[['configuration','OR_median','OR_lower','OR_upper']]
r = pd.read_csv(OUT/'e3_repeated_seed_metrics.csv')
r = r[(r.risk_budget==.2)&r.mask_group.isin(['all','missing_both'])]
rt = r.groupby(['model','features','policy','mask_group']).agg(auto_n_mean=('auto_n','mean'),auto_rate_mean=('auto_rate','mean'),risk_mean_served=('risk','mean')).reset_index()
rt.to_csv(OUT/'report_repeated_summary.csv',index=False)
s = pd.read_csv(OUT/'e2_boundary_summary.csv')
example = s[(s.n_total==30000)&np.isclose(s.double_fraction,.15)&np.isclose(s.target_complete_auc,.9)&np.isclose(s.missingness_strength,np.log(1.75))&np.isclose(s.budget,.1)&s.policy.isin(['G0','G2'])]
example = example[['mechanism','policy','auto_rate','risk_mean_served','double_auto_rate','double_risk_mean_served','double_service_fraction']]
fig,ax=plt.subplots(figsize=(10,5))
for i,row in a.iterrows():
    ax.errorbar(row.AUROC_mean,i,xerr=[[row.AUROC_mean-row.AUROC_min],[row.AUROC_max-row.AUROC_mean]],fmt='o',capsize=4)
ax.set_yticks(range(len(a)),a.model+' / '+a.comparison)
ax.axvline(0,color='gray',linestyle='--');ax.set_xlabel('Delta AUROC: mask minus baseline (5 seed range)')
fig.tight_layout();fig.savefig(FIG/'e1_mask_increment.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
labels=example.mechanism+' / '+example.policy
axs[0].barh(labels,example.double_auto_rate);axs[0].set_xlabel('Double-missing automatic service rate')
axs[1].barh(labels,example.double_risk_mean_served.fillna(0));axs[1].axvline(.1,color='red',linestyle='--');axs[1].set_xlabel('Exact population risk, mean over served runs')
for i,x in enumerate(example.double_risk_mean_served):
    if pd.isna(x): axs[1].text(.005,i,'NO SERVICE: risk undefined',va='center',fontsize=8)
fig.suptitle('Illustrative synthetic cases; N=30000, nominal AUC=.9; not patient results')
fig.tight_layout();fig.savefig(FIG/'e2_illustrative_boundary.png',dpi=180);plt.close(fig)
q=rt[(rt.features=='preop')&(rt.mask_group=='all')&rt.policy.isin(['G0','G1','G2'])]
fig,ax=plt.subplots(figsize=(7,4));ax.bar(q.model+' / '+q.policy,q.auto_rate_mean*100)
ax.set_ylabel('Automatic service (%)');ax.set_title('Real data: repeated CV, budget .20; mean across 5 seeds')
fig.tight_layout();fig.savefig(FIG/'e3_real_service.png',dpi=180);plt.close(fig)
primary=pd.read_csv(OUT/'e3_metrics.csv')
assert primary.auto_n.sum()==0
assert len(s)==14976 and s.case_id.nunique()==832
assert json.loads((OUT/'validation_v2.json').read_text())['status']=='PASS'

report = '''# Better-1：本轮实验结果与导师汇报依据

日期：2026-09-17。Material Passport：Origin Skill = experiment-agent；Origin Mode = run/validate；Version = selective_risk_v2；Verification Status = ANALYZED（本地运行及程序校验通过，尚未经独立复现）。

## 1. 当前结论

**建议收缩为“检查缺失下选择性风险控制的样本量与服务率边界”，暂不把“分块显著改善真实患者服务”作为主张。** 合成机制证明问题能够发生、方法在部分条件下能够起作用；当前真实队列尚未证明稳定增量价值，而且双缺失组分块认证后没有自动服务。不能为证明 idea 有潜力而只挑选有利结果。

此轮属于探索性研究。此前已经查看过同一队列；本次新划分不是全新的独立验证集。结果不能解释为临床部署安全性或跨中心泛化证明。

## 2. E0：数据与可用性

用户已明确确认：C42=1 表示本次手术病理证实的盆腔淋巴结阳性；术前模块在该手术前可得。内部数据复现 773 人、149 阳性：完整 401/76；仅缺 TCT 240/40；双缺失 113/33；仅缺 HPV 19/0（人数/阳性数）。19 人小组不独立认证。

本轮从同一患者身高体重重算 770 个可用 BMI，保留原始数据不动。HPV/TCT 原始值保留，只规范标点，不猜测复杂编码语义。异常 CA125 字符值转缺失并留痕。74 个重复 SOP 副本仅在派生影像索引标记排除，未删除文件。1590740 的 SE1 标为真实缺失。

E0 技术校验通过，临床语义冻结仍为部分完成：3 人影像日期晚于手术、75 人影像淋巴结字段跨来源不一致、部分检查编码和身高异常、DICOM 标识到研究患者的来源链接仍待核实。当前实验使用对齐表字段，并提供不含影像表格字段的分析；未训练影像网络。手术年份只作回顾性混杂调整，不能当成已确认可部署输入。无术后病理字段进入分数器。

## 3. 方法与验证范围

主风险为 R_auto=P(Y=1|AUTO_NEGATIVE)，服务率为 P(AUTO_NEGATIVE)；漏诊捕获比例 R_miss=P(AUTO_NEGATIVE|Y=1)。没有自动服务时 R_auto 未定义，不记作零。

固定 21 个阈值、6 个作用域，共 126 个候选，用单侧二项检验/Clopper–Pearson 上界配合 Bonferroni 置信预算，允许在该候选家族中选择阈值和分区。95% 同时性针对每个冻结模型与划分，不覆盖所有模型、种子或后续挑选出的结论。父块回退采用整分区退化，不能把父块证书当成原始子群证书。补查仅是建议，无临床收益估计。

主分析采用标签无关随机划分：拟合 386、校准 193、测试 194 人；另作 5 个种子×5 折稳健性分析。预处理只在拟合样本学习。普通未校正搜索仅作为失效对照；校正程序和锁定阈值独立认证作为有效对照。

## 4. E1：缺失模式是否提供稳定信息？

基线不包含 HPV/TCT 值或缺失指示，再添加 mask，避免把已隐含缺失信息的模型当作“无缺失信息”基线。以下是 5 种子均值及 AUROC 范围；Brier 差值越负越好。范围不是置信区间。

{e1}

双缺失相对完整组的惩罚逻辑回归 OR，来自每种配置 300 次完整重拟合 bootstrap（共 900 次）：

{or_table}

这些区间均包含 1；不能声称稳定独立关联或因果效应。仅缺 HPV 的 19 人零阳性存在分离，不能宣称该组具有保护作用。固定 OOF 预测上的条件 bootstrap 区间也都跨过 Brier 零差值，该区间不包含完整训练过程不确定性。

时间前推（2015–2018 训练 429 人，2019–2020 测试 344 人）：LR AUROC 0.6409→0.6467，但 Brier 0.15525→0.15543；梯度提升 AUROC 0.6284→0.6411，Brier 0.16068→0.15866，AUPRC 略降。年份和扫描仪调整后 LR AUROC 0.6412→0.6573，Brier 0.14513→0.14471。综合属于弱且不一致的增量信号。

![缺失模式增量](../results/selective_risk_v2/figures/e1_mask_increment.png)

## 5. E2：什么时候方法可能有价值？

完成 832 个配置×200 次独立抽样，共 166,400 个合成数据集，输出 2,995,200 条策略—预算重复记录。覆盖 MCAR、MAR、信息性缺失和评分信息损失情境；风险预算 0.05/0.10/0.20。总样本量 1,000/3,000/10,000/30,000，双缺失比例 0.05/0.15/0.30/0.50，完整组标称 AUROC 0.6/0.7/0.8/0.9。

这是具有可精确计算总体风险的离散分数分布实验，不是训练 166,400 个模型，也不是真实患者证据。MCAR 不强制风险比不同；缺失强度是 log-odds 参数，实际风险比和实际 AUROC 单独输出。信息损失情境仅为抽象代理，尚未执行完整协变量遮蔽与模型重训练实验。

下面是**结果出来后选取的解释性案例**：总人数 30,000、双缺失比例 0.15、完整组标称 AUROC 0.9、缺失强度 log(1.75)、预算 0.10。不能将有利案例当成全网格成功率。

{examples}

其中信息性缺失下，全局总体风险约 7.35%，双缺失风险却约 16.35%；G2 将双缺失风险降至约 4.32%，该组自动率约 31.43%。但是加入评分信息损失后，G2 双缺失组自动率为零。这说明是否有用同时依赖样本量与分数区分能力，增加人数并不必然解决问题。

未校正 G2 搜索的证书不覆盖概率在整个设计网格上平均约 10.16%，校正 G2 约 0.063%。这是不同固定配置上的 Monte Carlo 描述均值，不是临床频率；每格只有 200 次，不能以某一格的最大偏差直接断言理论失败，也不能用模拟代替证明。

![合成边界](../results/selective_risk_v2/figures/e2_illustrative_boundary.png)

## 6. E3：真实队列能留下多少服务？

**主随机划分下，四个分数器、所有预算和策略均未形成非空自动判阴通道。** 双缺失校准样本仅 34 人；在 126 候选校正下，即使零阳性，要让 20% 风险预算的上界通过也至少需要 36 个被自动接收的校准样本。因此该划分存在直接的样本量障碍，不能把零漏诊写成安全成功。

下表为风险预算 0.20、术前分数器的重复交叉验证结果。自动人数是每个种子内汇总后再跨 5 个种子平均；不是独立患者人数。risk_mean_served 是有服务种子的风险均值，不是合并比例，也不是认证上界。all 为全部患者，missing_both 为双缺失。

{real}

真实 G2 双缺失组在本轮重复实验中始终无自动服务；全局规则有时服务该组，不能由此声称该子群已得到风险保证。跨折经验风险波动或超过预算，也不等于单次规则的总体保证必然失效。

服务率代价可定义为同一划分、同一分数器、同一预算下 S_global−S_group（百分点）；只有两者服务率均有定义时比较，风险保证作用域同时注明。此处术前 LR 的平均总体服务率约 11.20%→3.00%，代价约 8.20 个百分点；梯度提升约 6.29%→1.27%，代价约 5.02 个百分点。减少服务并不自动等于子群获益，尤其该组完全不服务时。

![真实服务率](../results/selective_risk_v2/figures/e3_real_service.png)

## 7. 已做、未做与下一步

- 已完成：E0 技术审计、E1 真实关联/增量/时间与扫描仪敏感性、E2 评分分布边界实验、E3 主划分与重复交叉验证、选择校正对照、900 次重拟合 bootstrap。
- 校验通过：患者角色不重叠、动作唯一、动作与阈值一致、空服务风险缺省、二项检验与 CP 对偶、合成总体风险向量计算与直接计算一致、原始数据 SHA 未改变。校验由本工程执行，尚非独立复现。
- 尚未完成：临床异常值与影像时序最终裁定、原始协变量遮蔽机制重训练、更多同一风险定义下的强基线、MRI 网络、独立外部验证。标准预测集覆盖率方法不能直接冒充 R_auto 对照。
- 当前不进入大规模 MRI 或外部泛化宣称：真实主线的增量和服务率闸门尚未满足。下一步优先核对临床异常及标识链接，再以固定协议研究“多少校准样本、多少区分能力才能获得非零服务”的边界，避免反复调阈值追求正结果。
- 导师决策建议：认可协议/边界研究才继续；若必须证明当前队列的分块性能优势，则现有证据不足，应暂停该主张。补查收益仍不作主要创新。

偏差检查：用子群报告防 Simpson 掩盖；不从组间关联推个体因果（生态谬误）；手术入组涉及 Berkson/选择偏差；检查路径可能是 collider；显式报告基准患病率；重复划分仅检验波动，不能消除回归均值；手术队列存在幸存与入组筛选；候选多重校正应对 look-elsewhere；冻结协议及完整失败结果减少 forking paths；关联不解释为因果；时间窗确认及影像异常核对用于减少反向因果。以上不等于所有偏差已消除。

## 8. 可复现入口与材料

执行协议：`reports/v2_execution_protocol.md`。输出：`results/selective_risk_v2/`。依次使用本地 pytorch 环境运行 `src/v2_pipeline.py`、`src/synthetic_boundary_v2.py`、`src/validate_v2.py`、`src/v2_robustness.py`、`src/report_v2.py`；重新运行会更新此版本目录，请先保存待对比版本。所有 *_LOCAL 文件包含患者级研究标识，只在本地保留，不宜直接作为公开附件。

本轮以 CPU 表格与合成实验为主，不依赖 4GB GPU；不能据此估算完整 3D MRI 训练的资源需求。
'''.format(e1=table(a),or_table=table(b),examples=table(example),real=table(rt[(rt.features=='preop')&rt.policy.isin(['G0','G2'])]))
path=ROOT/'reports/Better1_v2_实验推进报告_20260917.md'
path.write_text(report,encoding='utf-8')
(OUT/'report_validation.json').write_text(json.dumps({'status':'PASS','summary_rows':len(s),'scenarios':int(s.case_id.nunique()),'primary_all_zero_service':True,'figures':3},indent=2),encoding='utf-8')
print(path)


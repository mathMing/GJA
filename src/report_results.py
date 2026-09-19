import os, pandas as pd, numpy as np
from common import build_parser,load_config,resolve_dirs
A=build_parser('report_results','Better-1 advisor report').parse_args();C=load_config(A.config);OUT,TMP=resolve_dirs(C,A)
def rd(n):
 p=os.path.join(OUT,n);return pd.read_csv(p) if os.path.exists(p) else None
s,m,acts=rd('data_audit_summary.csv'),rd('three_way_model_summary.csv'),rd('three_way_actions.csv')
if m is not None:
 m.to_csv(os.path.join(OUT,'risk_service_curve.csv'),index=False,encoding='utf-8-sig')
if acts is not None:
 q=acts.groupby(['model','policy','risk_budget','mask_group','action'],as_index=False).agg(n=('patient_id','size'),positives=('true_label','sum'))
 q.to_csv(os.path.join(OUT,'mask_risk_summary.csv'),index=False,encoding='utf-8-sig')
 try:
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(7,5))
  for (model,policy),z in m.groupby(['model','policy']):
   ax.plot(z.risk_budget,z.auto_rate,marker='o',label=f'{model}/{policy}')
  ax.set(xlabel='Risk budget',ylabel='Auto-negative rate',title='Risk-service curve');ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(os.path.join(OUT,'risk_service_curve.png'),dpi=180);plt.close(fig)
 except Exception as e:
  open(os.path.join(OUT,'plot_error.txt'),'w').write(str(e))
with open(os.path.join(OUT,'advisor_summary.md'),'w',encoding='utf-8') as f:
 f.write('# Better-1 最小可行性验证汇报摘要\n\n')
 f.write('## 研究问题\n验证 HPV/TCT 路径缺失是否使全局自动判阴风险在特定子群失控，以及分块和回退是否保留可用自动服务率。\n\n')
 if s is not None:f.write('## 真实数据\n\n'+s.to_string(index=False)+'\n\n')
 if m is not None:f.write('## 内部三出路结果\n\n'+m.to_string(index=False)+'\n\n')
 f.write('## 初步判定\n\n')
 if m is not None and float(m.auto_rate.max())>0:
  f.write('当前结果显示严格风险预算下分块策略自动率显著受限；这支持“安全认证存在服务率代价”的研究问题，但尚不足以支持“分块规则临床可用”的结论。下一步应先检查选择后校正和分块回退设计，再决定是否继续复杂模型。\n\n')
 f.write('## 需要导师拍板\n1. 是否认可风险控制协议为主贡献；2. 是否认可 P(Y=1|AUTO_NEGATIVE) 为第一主终点；3. 是否接受安全—服务率权衡；4. 是否在修正分块校正后仍有非零服务率时申请独立外部数据。\n')
print('[DONE] report',OUT)


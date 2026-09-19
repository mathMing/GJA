import os, json
import numpy as np, pandas as pd
from common import build_parser,load_config,require_file,resolve_dirs,resolve_source
A=build_parser('data_audit','Better-1 E0 audit').parse_args();C=load_config(A.config);OUT,TMP=resolve_dirs(C,A);SRC=require_file(resolve_source(C,A))
def miss(s):
 t=s.astype('object').where(s.notna(),np.nan);return t.isna()|t.astype(str).str.strip().isin(['','NA','na','N/A','nan','None','无'])
raw=pd.read_excel(SRC,header=None);df=raw.iloc[2:].reset_index(drop=True);df.columns=[f'C{i:02d}' for i in range(df.shape[1])]
y=pd.to_numeric(df.C42,errors='coerce').astype(int);hpv,tct=miss(df.C13),miss(df.C14)
g=np.full(len(df),'complete',object);g[tct&~hpv]='missing_tct';g[tct&hpv]='missing_both';g[~tct&hpv]='missing_hpv'
rows=[]
for z in ['complete','missing_tct','missing_both','missing_hpv']:
 m=g==z;rows.append({'mask_group':z,'n':int(m.sum()),'positives':int(y[m].sum()),'prevalence':float(y[m].mean())})
pd.DataFrame(rows).to_csv(os.path.join(OUT,'data_audit_summary.csv'),index=False,encoding='utf-8-sig')
pre=[4,5,6,7,8,9,10,11,13,14,15,16,17,19,21,22,23,24,25];post=list(range(34,46))+list(range(47,62));leak=[]
for j in range(df.shape[1]):
 m=~miss(df[f'C{j:02d}']);leak.append({'column':f'C{j:02d}','usable_preop':j in pre,'blacklisted_postoutcome':j in post,'observed_n':int(m.sum()),'observed_positive_n':int(y[m].sum())})
pd.DataFrame(leak).to_csv(os.path.join(OUT,'data_leakage_audit.csv'),index=False,encoding='utf-8-sig')
with open(os.path.join(OUT,'data_flow.md'),'w',encoding='utf-8') as f:f.write(f'''# Better-1 数据流审计

- 真源：{SRC}
- 样本：{len(df)}；阳性：{int(y.sum())}
- 分块键：HPV(C13) × TCT(C14)
- 术前白名单：{len(pre)} 列；术后/结局后黑名单：{len(post)} 列
- 1590740 的序列级缺失由影像管线单独记录，不进入本表特征。
- 仅缺 HPV（19例）不单独认证，回退到缺任一父块。
''')
print('[DONE] audit',OUT)


"""Build the read-only data audit report from independently collected evidence."""
import json,hashlib,datetime
from pathlib import Path
import pandas as pd
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917")
OUT=Path("X:/GJA/WORK/reports/DATA数据审计报告_20260917.md");OUT.parent.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((O/n).read_text(encoding="utf-8"))
c=read("clinical_audit.json");x=read("cross_audit.json");u=read("supplement.json");im=read("image_audit.json")
ims=read("image_supplement.json")
inv=pd.read_csv(O/"file_inventory_LOCAL.csv");p=pd.read_csv(O/"field_profile.csv").fillna("")
def table(headers,rows):
 def fmt(v):return str(v).replace("|"," / ").replace("\n"," ")
 return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(fmt(v) for v in r)+" |" for r in rows)
versions=table(["版本","数据行","唯一住院号","阳性数","说明"],[[v["book"],v["n"],v["id_unique"],v["label_counts"].get("1",0),"主分析源" if v["book"]=="cervical_master_aligned.xlsx" else "备份/来源，不混合入组"] for v in c["versions"]])
groups=table(["模式","人数","阳性","阳性率"],[[v["group"],v["n"],v["positive"],f'{100*v["prevalence"]:.2f}%'] for v in c["groups"]])
num=table(["字段","零值数","最小值","中位数","最大值"],[[k,v["zero_n"],v["min"],v["median"],v["max"]] for k,v in u["numeric_profiles"].items()])
whitelist={4,5,6,8,9,10,11,13,14,15,16,17,19,21,22,23,24}
def disposition(i):
 if i==42:return "仅标签，禁止输入"
 if i in [0,1,2,3]:return "标识/备注，禁止输入"
 if i==7:return "先由同患者身高体重重建"
 if i==25:return "自由文本，先排除并核实时间"
 if i in whitelist:return "术前候选；需确认时点及编码"
 if i>=27:return "术中/术后/随访或空列，禁止输入"
 return "空列，不输入"
fieldtable=table(["索引","Excel列","字段","有值","缺失","处理"],[[v.column,v.excel_column,v.header,int(v.nonmissing),int(v.missing),disposition(int(v.column[1:]))] for v in p.itertuples()])
sources=[]
for name,h in c["source_sha256"].items():
 path=next((R/"clinical").rglob(name));now=hashlib.sha256(path.read_bytes()).hexdigest()
 sources.append([str(path),h,now==h])
 assert now==h,"Source workbook changed during audit"
manifest=read("../ars_validated_v1/manifest_all.json") if False else json.loads(Path("X:/GJA/WORK/results/ars_validated_v1/manifest_all.json").read_text())
source_match=manifest["source_sha256"]==c["source_sha256"]["cervical_master_aligned.xlsx"]
zs=inv[inv.bytes==0].path.tolist()
desc=pd.DataFrame(im["series_descriptions"])
desctable=table(["中心","文件夹序列","DICOM描述","文件数"],desc.values.tolist())
series=pd.read_csv(O/"series_audit_LOCAL.csv",dtype={"patient_id":str}).fillna("")
mismatch=pd.read_csv(O/"dicom_identity_mismatch_LOCAL.csv",dtype=str)
# A technical inconsistency is a review flag, not automatic corruption.
tech=table(["头信息检查","结果"],[
["DICOM文件全量解析",f'{im["dicom_total"]}；失败 {im["read_errors"]}'],
["有DICOM数据的目录序列数",im["series"]],
["患者ID原样不等于目录名的文件",im["id_literal_mismatch_files"]],
["仅去除前导零后仍不匹配的文件",im["id_zero_normalized_mismatch_files"]],
["重复SOP Instance UID记录数",im["duplicate_sop_rows"]],
["一个序列目录包含多个Series UID",im["series_multiple_uid"]],
["一个序列目录内矩阵大小变化",im["series_multiple_shape"]],
["一个序列目录内方向值变化",im["series_multiple_orientation"]],
["有重复空间位置的序列目录",im["series_duplicate_position"]]])
years=pd.read_csv(O/"missingness_by_surgery_year.csv").pivot(index="year",columns="group",values="n").fillna(0).astype(int)
yearstable=table(["手术年"]+list(years.columns),[[idx]+row.tolist() for idx,row in years.iterrows()])
report=f"""# X:\\GJA\\DATA 数据审计报告

审计日期：2026-09-17  
目的：核对研究数据的真实口径、技术完整性、字段可用性和对当前实验的影响。  
方式：原始文件只读；本报告及审计记录写入 WORK。没有修改原表、重算保存原工作簿、修复影像或重训模型。  
证据等级：文件结构和程序化一致性检查；不替代临床源记录与影像/病理医生复核。

## 一、结论先行

**当前数据可以继续用于受限的内部探索，但不能按“已清洗、已冻结、可直接用于外部验证”的数据包使用。**

通过的关键核对：

- 内部 aligned 主表 **773 人、149 例盆腔淋巴结阳性**；住院号唯一，与内部影像目录一一对应。
- HPV/TCT 四格人数及阳性数与此前实验一致。
- 丽水影像对应 **92 人、16 例阳性**，可从临床表唯一匹配。
- 当前 aligned 文件与此前正式实验清单中的源文件 SHA256 一致：**{source_match}**。本次新发现并非因为另换了一份 aligned 表。
- 头信息及文件可读性检查的完整结果见第七、八节；“文件可读”不等于临床内容正确。

需要优先处理的问题：

| 优先级 | 发现 | 对研究的影响 |
|---|---|---|
| P0 | aligned 有 770 个 BMI 公式，720 个行引用错位，所有公式缓存为空 | 不能把 BMI 缺失当作临床缺失，也不能直接刷新原公式 |
| P0 | 丽水目录的 clinical_stats.xls 含三个中心，包含全部 773 名内部患者 | 整张表不能充当外部验证集，需按中心与患者索引限定 |
| P0 | 术后 C34 的是否有值与标签完全一致 | 任意使用该字段或缺失指示都会形成结局泄漏 |
| P0 | 3例DICOM检查日晚于表格手术日；78例为同日 | MRI术前预测必须先核对实际时间，不能自动认定影像为术前 |
| P1 | 内部表 C24 与混合中心表 E16 在同一批患者中有 75 例不同 | 影像淋巴结状态定义不可直接视为一致，需源报告判定 |
| P1 | 主源 master 有重复住院号；aligned 的辅助表未同步缩至 773 人 | 必须用明确主表和患者键，不能按辅助表行号拼接 |
| P1 | 两条日期为 Excel 数字；一条 CA125 为“·3.9”；HPV/TCT 编码不统一 | 清洗必须保留原值和转换记录，不能直接强制转数值 |
| P1 | 病理图像覆盖仅 214 人，其中 115 阳性 | 是明显富集阳性的子集，不是随机验证样本 |
| P1 | 一个内部患者目录含74组字节完全相同的DICOM副本 | 需在派生序列中去重，避免重复堆叠切片 |

P0/P1 是本报告的处理优先级，不是对临床事件的严重程度分级。

## 二、审计范围和目录盘点

共 **{len(inv):,} 个文件，{inv.bytes.sum()/1e9:.3f} GB（十进制）**：

| 内容 | 数量 |
|---|---:|
| 临床工作簿 | 5（4 个 xlsx，1 个 xls） |
| 内部无扩展名 DICOM | 83,342 |
| 丽水 .dcm | 13,549 |
| 病理 JPG | 434 |
| 文本说明 | 2 |

全目录清单包含路径、文件大小和修改时间。零字节文件：{len(zs)} 个，具体为 {', '.join(zs)}；应结合文件类型解释，不将说明文件自动判作影像损坏。

检查覆盖：

1. 五个临床工作簿的全部工作表结构；内部主表 62 列的缺失、类型、取值和标签逻辑。
2. 内部三个版本按住院号比对；重复键单独保留，不任取一条当真值。
3. 内部/丽水表格、MRI 目录、病理目录的患者匹配。
4. 全部 {im["dicom_total"]:,} 个 MRI 文件的 DICOM 头信息；不解码全部 MRI 像素。
5. 全部 434 个 JPG 解码及文件哈希；未完成 OCR 或逐份病理诊断核验。
6. 原始工作簿起止 SHA256 验证，均未改变。

## 三、内部队列与版本口径

{versions}

本次实测 master 是 **789 行、788 个唯一住院号、150 个阳性行**；backup 是 **770 人、148 阳性**。此前旧审计材料中的 master“154 阳性”和 backup“149 阳性”与当前源文件不符，应更正。该更正不改变 aligned 的 773/149 主实验口径。

master 相比 aligned 多出 15 个唯一住院号、16 条记录，其中包含一个重复住院号。该重复患者的两条记录手术日期及部分内容不同，不能简单按重复行删除，也不能当成两个独立患者；当前 aligned 未包含该住院号。

backup 是 aligned 的患者子集，缺少 3 人；共同患者的主表单元格内容在规范化比较中一致。

### 工作表不能混用

- aligned Sheet1：两行表头，773 条患者记录，是本次研究主表。
- Sheet2：759 行结构、739 个非空行，属于辅助内容，未形成与773人对应的标准患者表。
- Sheet3：790 行结构，含789行辅助计算内容；502行隐藏。
- Sheet2/Sheet3在三个版本中保留旧规模，不能直接与aligned Sheet1按行位置拼接。
- 所有字段定位使用明确工作表、列号和患者键；不得依赖当前显示行序。

### 结局定义

C42（Excel AQ列）是**盆腔淋巴结状态**，不是“是否宫颈癌”。因此本研究的“自动判阴”必须解释为对这一结局的判断，不能写成排除宫颈癌。

773条标签均为0/1，无缺失。标签与数值可解析的转移淋巴结个数之间，未发现“标签0但个数>0”“标签1但个数0”矛盾；没有转移个数超过切除总数的记录。但转移个数有1条公式缓存缺失，因此不能宣称全部标签已得到独立病理验证。

## 四、字段质量、缺失与泄漏

### HPV/TCT 四格

{groups}

两列的缺失主要以字符串 NA 保存。本次将空白/NA等视为缺失；阴性、NILM、数值0等不自动当缺失。

这些字段证明的是“表格是否记录结果”，不能直接证明患者未做检查，更不能证明缺失由特定诊疗路径造成。缺失原因和采集时间需要源记录补充。

编码问题：

- HPV混有单个分型数字、多分型字符串、阳性/阴性/未分型；例如“16.33”“31.52”和“16、、58”需核实。分型数字不能当连续数值使用。
- TCT有ASC-US/ASCUS等不同写法，以及复合结论和“建议活检”等非标准结果。需保留原值，由临床确认合并映射，不应直接转数字或任意排序。
- 仅缺HPV组19人、零阳性只是观察结果，不能证明该组风险为零。

缺失模式随手术年份变化：

{yearstable}

例如双缺失在2015年为25/95，在2020年为18/172。手术年份不是检查日期；这种时间相关性提示应考虑年份/收集流程混杂，不能直接归因于临床风险驱动缺失。

### BMI与公式错误

- aligned主表770个BMI公式，其中720个引用其他行，16个引用超出当前数据行范围。
- 示例：H53使用 `=F54/(G54%*G54%)`，不是同一行的F53/G53。
- 主表还有J3的 `=J3:L930` 和AS619的 `=-AR6261`，包含自引用/范围及越界问题；这些异常表达式在master源文件中也存在，不全是aligned阶段新增。
- aligned主表772个公式、Sheet2的1473个公式、Sheet3的789个公式缓存均为空。直接读值时会出现技术性缺失。
- BMI仅3个可读值不能解释为770名患者未测BMI；其中770名患者的身高和体重同时可解析，可另行计算BMI，但需先核对单位及异常原值。
- **不要对现有错误公式直接执行“全部重新计算”作为修复。** 应在派生数据中按同一患者重建，经核对后冻结。

### 数值与日期

{num}

以上为原值范围，不将极端值自动认定为错误。需要优先核实：

- 身高最小112 cm：确认原始记录，不能直接改为其他数值。
- SCCA有2个零值；B超肿瘤直径有130个零值；MRI肿瘤大小有46个零值。需区分未见病灶、无法测量和未记录，不能把0统一当真阴性或缺失。
- C17（CA125）有一条文本“·3.9”（中点U+00B7），不是已确认的“<3.9”；不得猜测比较符号。此前实验按缺失处理，应保留该例核对记录。
- 手术日期771条为日期类型，2条为Excel序列数43543、43629。按Excel日期原点转换后，范围为2015-01-07至2020-12-25。直接用默认时间解析会错误得到1970年。
- 主表另有一个绝经状态和一个转移淋巴结个数因公式缓存为空而不可读。

### 泄漏字段

C34“除淋巴结阳性外临床病理分期”有值149人，全部为阳性；其是否有值与C42标签完全对应。C45“转移淋巴结部位”有值144人，全部为阳性。

因此术后病理字段、相关缺失指示、术中记录、治疗、随访及标识均不得进入术前预测输入。字段叫“分期”并不代表它是术前可得。

C24来自“术前MRI”表头模块，暂作候选输入，但表头不能证明每条记录的实际采集时点；详细描述C25也需逐条确认是否混入术后结论。当前实验未使用C25。

随访及辅助治疗字段几乎为空，不能据此直接扩展生存、复发或治疗效果研究。

## 五、丽水与混合中心临床表

`clinical/lishui/clinical_stats.xls`实际有 **1027条记录**：

| 来源标记 | 人数 |
|---|---:|
| 温医 | 788 |
| 宁一 | 137 |
| 丽水 | 102 |

其中全部773名aligned内部患者均能在该表找到，标签完全一致；这不构成独立标签复核，因为可能来自同一数据来源。

**必须先按中心限定，再用规范化患者键匹配，不能把该文件整体称为“丽水外部队列”。**

丽水患者索引文件没有标题行，共102个唯一编号。影像目录92人，均在索引内，另10人没有当前影像目录。92人均唯一匹配丽水临床记录，其中16阳性、76阴性。

丽水文件夹保留前导零，而Excel编号为数字。匹配时需保留原编号，并使用中心+规范化编号作为连接键；没有发现当前内部与丽水影像患者数值编号重叠，但这不足以排除跨中心同一人的可能性。

### 跨文件字段不一致

内部C24“淋巴结是否阳性（按长径）”与混合中心表E16“影像学>=1cm/提示转移瘤”在共同773人中有75例不一致。

这是需要核实的定义/来源差异，不自动判哪一张表错误。若把两列直接当同一特征，会造成训练和外部评价输入口径不一致。年龄、孕产次、可解析化验值及标签的对应比较未出现数值冲突。

丽水表无HPV/TCT字段，不能执行同样的缺失模式条件规则。术前特征组合也不完全一致，不能仅冻结模型权重就宣称能直接外部验证。E19宫颈浸润深度在匹配92人中全部缺失。

宁一137人是额外表格来源，但当前DATA目录没有相应MRI目录，不能视为现成的独立MRI验证队列。

## 六、病理图像覆盖

共434张JPG，214个患者目录，全部属于773名内部患者。解码失败 **{im["pathology_decode_errors"]}**，字节完全相同的重复图像记录数 **{im["pathology_exact_duplicate_rows"]}**。

覆盖人群为115阳性、99阴性，阳性比例53.7%，高于全队列19.3%。阳性患者的覆盖率为115/149=77.2%，阴性患者为99/624=15.9%。因此病理图像是否存在本身与标签相关，不能当作术前输入或无偏随机抽样指标。

附带说明提到“总共214例、缺1例”及部分图像模糊；当前实际214个有JPG的目录，缺失所指患者及是否已补齐无法仅据此确认，需要交付清单。文件可解码不代表图像文字/组织细节足以诊断。

本次未逐图视觉审阅或OCR，不将其认定为可直接训练的病理切片数据，也未依据照片重新判定C42标签。

## 七、MRI完整性与技术核对

内部773人：772人有SE0/SE1/SE2数据，1590740的SE1只有“缺T1.txt”，无DICOM。该缺失必须明确记录，不得用零影像伪装成真实T1序列。丽水92人均有三个序列目录及数据。

{tech}

序列命名须依DICOM元数据核实，SE0/SE1/SE2文件夹编号本身不保证不同中心或患者具有同一医学序列含义。

身份核对：所有DICOM PatientID均不同于文件夹住院号，但865个患者目录与各自中心内的PatientID为一一对应，未发现一号多目录或一目录多号，也未发现Study UID/Series UID跨患者目录复用。这更符合不同标识体系的表现，不能直接判定全部错配；仍需原始影像号—住院号映射证明身份正确。

重复核对：74个重复SOP UID的两份文件均经SHA256确认为字节相同，全部位于内部患者1588646的三个序列目录，没有跨患者重复SOP。应在派生数据中保留一份并记录来源，不修改原始归档。未进行不同UID的全像素去重，因此不能排除所有内容重复。

时序核对：773名内部患者均有可解析StudyDate。3人检查日晚于手术日（最晚18天），78人为同日，日期层面无法判断先后；检查至手术最大间隔224天，中位数2天。后于手术的记录应先隔离为待核对，确认是日期录入错误还是术后影像，不能直接作为术前MRI输入。具体记录见imaging_surgery_timing_LOCAL.csv。

跨中心差异：内部扫描日期2015-01-05至2020-12-21，丽水2021-06-03至2024-02-27；内部主要Philips/GE、3T，丽水主要SIEMENS、1.5T，另有Philips/UIH。丽水SE1含动态/Dixon等描述，不能仅凭SE1统一视为同一种T1协议。设备和时间迁移同时存在，外部性能变化不能单独归因于缺失模式。

{desctable}

技术解释：

- 同一序列目录的多个UID、方向或矩阵变化需要逐项核对；不能未经拆分就强行堆成一个三维体积。
- 同一空间位置重复可能来自扩散多b值/多次采集，不直接视为重复坏片。
- PatientID与文件夹差异应区分前导零与实质不匹配；不匹配病例详单留在本地。
- 本次读取所有头信息，不保证全部像素可解码，也不保证无运动伪影、无扫描缺层、无误分序列。未做临床阅片、分割质量评估或全部像素内容去重。
- 没有在DATA盘点中发现单独分割标注文件；不能据此排除标注存放在其他目录。
- 上述日期比较基于StudyDate和表格手术日期，不替代手术时刻、真实序列采集时刻及源记录核验。

## 八、对已完成实验的影响

1. **主队列人数和HPV/TCT分组可信**：773/149及四格计数已复算，当前源文件哈希与正式实验一致。
2. **已有实验未使用BMI，因此BMI错位不直接进入那轮模型。** 但若以后恢复BMI或加入通用缺失指示，必须先修复派生数据。
3. 绝经状态、转移个数等的技术缺失需要记录；后者虽不作为输入，也影响标签核验的完整性。
4. 当前“四个分数器”的比较是在尚未完全标准化的HPV/TCT编码上进行；不能将现有性能视为充分清洗后的能力上限。
5. “路径驱动缺失”仍未得到数据来源证明，建议使用“记录的检查缺失模式”。
6. 现有内部人群为有手术/病理结局的选择性队列，不能直接推广到一般筛查人群。
7. 丽水没有缺失分块键，且影像状态字段口径存在不一致，外部验证前必须另定兼容输入及支持不匹配策略。
8. 旧审计文档中master/backup阳性数应更正；“无泄漏”“真源已冻结”等表述需收窄为具体通过的检查。

## 九、建议整改顺序

### 第一优先：形成可追溯派生表

- 原始工作簿只读保留，以中心+患者ID建立唯一键，生成版本和入排记录。
- 不依赖原公式缓存；BMI从同患者身高体重重建，同时标注不可计算及需核对病例。
- 分别处理J3、AS619异常公式，依据原始记录或经确认的权威来源补录，不能仅凭旧缓存猜值。
- 统一Excel日期转换，保留原值、转换值和转换理由。
- 为CA125“·3.9”、HPV多型编码、TCT类别及肿瘤尺寸0制定明确规则，未确认值继续标为待核对。

### 第二优先：标签与跨中心定义核验

- 由临床人员确认C42结局、时间窗口和病理依据。
- 审核75例影像阳性字段差异，建立C24/E16映射，不先假设可互换。
- 明确缺失原因、录入时点、术前影像解读时点。
- 病理图像仅在其覆盖子集开展标签核对，报告选择偏倚与无法核实的其余患者。

### 第三优先：影像训练前质量控制

- 明确1590740缺失序列的处理策略。
- 先复核3例检查日晚于手术日及78例同日记录；保留真实时点证据。
- 对1588646的74组重复SOP在派生序列中去重，保留完整日志。
- 依据UID、扫描协议、方向、矩阵、空间位置区分真实序列；复核所有头信息异常。
- 完成像素解码、代表性阅片和配准/分割检查后，再冻结MRI预处理。
- 数据清洗发生变化后重新运行受影响实验，旧结果作为历史版本保留。

本次只完成审计与问题定位，未执行上述数据修复；这避免把未经临床确认的判断写回源文件。

## 附录A：62列字段审计与准入建议

列号C00–C61为0-based程序索引；Excel列为工作簿实际定位。缺失计数按读取到的值计算，包含公式缓存缺失，不能直接等同临床未测。

{fieldtable}

## 附录B：来源完整性

下表记录本次检查前后SHA256一致性；不包含MRI逐文件内容哈希。

{table(["源文件","SHA256","审计前后相同"],sources)}

## 附录C：本地审计证据

辅助文件位于 `X:\\GJA\\WORK\\results\\data_audit_20260917`：

- `field_profile.csv`：逐字段类型、缺失、数值范围和阳性分布。
- `formula_audit.csv`、`formula_references_LOCAL.csv`：公式缓存及行引用。
- `version_cell_differences_LOCAL.csv`、`clinical_issues_LOCAL.csv`：版本与具体待核对记录。
- `cross_audit.json`、`internal_external_table_comparison.csv`：跨文件/中心匹配。
- `dicom_headers_LOCAL.csv`、`series_audit_LOCAL.csv`：全量DICOM头及序列汇总。
- `dicom_identity_mismatch_LOCAL.csv`、`duplicate_sop_LOCAL.csv`：影像身份/UID问题清单。
- `pathology_files_LOCAL.csv`：病理JPG解码及哈希。
- `missingness_by_surgery_year.csv`：年度缺失模式计数。
- `file_inventory_LOCAL.csv`：全量文件盘点。

带LOCAL的辅助文件含患者标识或具体记录，仅供本地核对。本报告不列姓名、电话号码或患者病理全文。

复现脚本位于WORK/src：data_directory_discovery.py、data_clinical_audit.py、data_cross_audit.py、data_audit_supplement.py、data_image_audit.py、build_data_audit_report.py。读取依赖使用本地隔离的pydicom/xlrd及捆绑Python；未改变训练环境。
"""
OUT.write_text(report,encoding="utf-8")
(O/"report_checks.json").write_text(json.dumps({"source_hashes_unchanged":True,"experiment_source_hash_matches":source_match,"report":str(OUT),"dicom_headers_read":im["dicom_total"]},indent=2),encoding="utf-8")
print(str(OUT))

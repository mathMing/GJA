import sys,json,hashlib,collections,time,warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"X:/GJA/WORK/.audit_deps")
import pandas as pd,numpy as np,pydicom
from PIL import Image
warnings.filterwarnings("ignore",category=UserWarning,module="pydicom")
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917")
f=pd.read_csv(O/"file_inventory_LOCAL.csv",keep_default_na=False)
images=f[f.path.str.startswith("raw/")].copy()
dc=images[~images.path.str.endswith(".txt")]
tags=["PatientID","StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID","Modality","StudyDate","SeriesDate","AcquisitionDate","SeriesDescription","ProtocolName","Manufacturer","MagneticFieldStrength","Rows","Columns","NumberOfFrames","InstanceNumber","ImagePositionPatient","ImageOrientationPatient","PixelSpacing","SliceThickness","SpacingBetweenSlices","EchoTime","RepetitionTime","BitsAllocated","BitsStored"]
def readrow(rel):
 parts=rel.split("/");row={"path":rel,"center":parts[1],"folder_patient":parts[3],"folder_series":parts[4]}
 try:
  ds=pydicom.dcmread(R/rel,stop_before_pixels=True,specific_tags=tags)
  for key in tags:
   value=getattr(ds,key,None)
   row[key]=str(value) if value is not None else ""
  row["transfer_syntax"]=str(getattr(ds.file_meta,"TransferSyntaxUID",""))
  pid=row["PatientID"]
  row["patient_id_literal_match"]=pid==parts[3]
  row["patient_id_zero_normalized_match"]=pid.lstrip("0")==parts[3].lstrip("0") if pid else False
 except Exception as ex:row["read_error"]=type(ex).__name__+": "+str(ex)[:120]
 return row
start=time.time();results=[]
with ThreadPoolExecutor(max_workers=6) as pool:
 for i,row in enumerate(pool.map(readrow,dc.path.tolist(),chunksize=32)):
  results.append(row)
  if (i+1)%10000==0:print("DICOM",i+1,"/",len(dc),round(time.time()-start,1),flush=True)
d=pd.DataFrame(results);d.to_csv(O/"dicom_headers_LOCAL.csv",index=False)
series=[]
for (center,pid,se),t in d.groupby(["center","folder_patient","folder_series"]):
 rec=dict(center=center,patient_id=pid,series=se,files=len(t),read_errors=int(t.get("read_error",pd.Series(index=t.index,dtype=object)).notna().sum()))
 for key in ["StudyInstanceUID","SeriesInstanceUID","PatientID","Rows","Columns","PixelSpacing","ImageOrientationPatient","StudyDate","SeriesDescription","EchoTime","RepetitionTime"]:
  rec[key+"_unique"]=t[key].nunique()
  rec[key+"_values"]=" | ".join(sorted(t[key].dropna().unique())[:6])
 rec["sop_duplicate_rows"]=int(t.SOPInstanceUID.duplicated(False).sum())
 rec["duplicate_position_rows"]=int(t.ImagePositionPatient[t.ImagePositionPatient!=""].duplicated(False).sum())
 series.append(rec)
s=pd.DataFrame(series);s.to_csv(O/"series_audit_LOCAL.csv",index=False)
patient=[]
for (center,pid),t in d.groupby(["center","folder_patient"]):
 patient.append(dict(center=center,patient_id=pid,n_files=len(t),series="|".join(sorted(t.folder_series.unique())),study_uids=t.StudyInstanceUID.nunique(),metadata_patient_ids=t.PatientID.nunique()))
pd.DataFrame(patient).to_csv(O/"image_patient_index_LOCAL.csv",index=False)
dup=d[d.SOPInstanceUID.ne("") & d.SOPInstanceUID.duplicated(False)]
dup.to_csv(O/"duplicate_sop_LOCAL.csv",index=False)
bad=d[~d.patient_id_zero_normalized_match.fillna(False)]
bad.to_csv(O/"dicom_identity_mismatch_LOCAL.csv",index=False)
# All pathology JPG images are decoded, no OCR or clinical adjudication.
jpegs=[]
for rel in f[f.path.str.endswith(".jpg")].path:
 rr=dict(path=rel,patient_id=rel.split("/")[2])
 try:
  with Image.open(R/rel) as im:
   im.load();rr.update(width=im.width,height=im.height,mode=im.mode)
  rr["sha256"]=hashlib.sha256((R/rel).read_bytes()).hexdigest()
 except Exception as ex:rr["error"]=str(ex)
 jpegs.append(rr)
j=pd.DataFrame(jpegs);j.to_csv(O/"pathology_files_LOCAL.csv",index=False)
summary={"dicom_total":len(d),"read_errors":int(d.get("read_error",pd.Series(dtype=object)).notna().sum()),"series":len(s),"patients_by_center":pd.DataFrame(patient).groupby("center").size().to_dict(),
"id_literal_mismatch_files":int((~d.patient_id_literal_match.fillna(False)).sum()),"id_zero_normalized_mismatch_files":len(bad),"duplicate_sop_rows":len(dup),"duplicate_sop_uids":dup.SOPInstanceUID.nunique(),
"series_multiple_uid":int((s.SeriesInstanceUID_unique>1).sum()),"series_multiple_shape":int(((s.Rows_unique>1)|(s.Columns_unique>1)).sum()),
"series_multiple_orientation":int((s.ImageOrientationPatient_unique>1).sum()),"series_duplicate_position":int((s.duplicate_position_rows>0).sum()),
"pathology_files":len(j),"pathology_patients":j.patient_id.nunique(),"pathology_decode_errors":int(j.get("error",pd.Series(dtype=object)).notna().sum()),"pathology_exact_duplicate_rows":int(j.sha256.duplicated(False).sum()),
"elapsed_seconds":time.time()-start}
for key in ["Modality","Manufacturer","MagneticFieldStrength","transfer_syntax"]:
 summary[key]=d.groupby("center")[key].value_counts().rename("n").reset_index().to_dict("records")
summary["series_descriptions"]=d.groupby(["center","folder_series","SeriesDescription"]).size().rename("files").reset_index().to_dict("records")
(O/"image_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k not in ["series_descriptions"]},ensure_ascii=True),flush=True)


import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917")
d=pd.read_csv(O/"dicom_headers_LOCAL.csv",dtype=str,keep_default_na=False)
du=d[d.SOPInstanceUID.duplicated(False)&d.SOPInstanceUID.ne("")].copy()
du["sha256"]=[hashlib.sha256((R/x).read_bytes()).hexdigest() for x in du.path]
du.to_csv(O/"duplicate_sop_hashes_LOCAL.csv",index=False)
same=du.groupby("SOPInstanceUID").sha256.nunique()
sd=pd.read_csv(O/"surgery_dates_LOCAL.csv",dtype={"patient_id":str})
sd["surgery_date"]=pd.to_datetime(sd.surgery_date)
it=d[d.center=="original"][["folder_patient","StudyDate","StudyInstanceUID"]].drop_duplicates()
it["patient_id"]=it.folder_patient.str.lstrip("0")
it["study_date"]=pd.to_datetime(it.StudyDate,format="%Y%m%d",errors="coerce")
m=it.merge(sd,on="patient_id",validate="many_to_one")
m["days_before_surgery"]=(m.surgery_date-m.study_date).dt.days
m.to_csv(O/"imaging_surgery_timing_LOCAL.csv",index=False)
r={"duplicate_sop_uids":len(same),"duplicate_uids_byte_identical":int(same.eq(1).sum()),"duplicate_uids_different_bytes":int(same.gt(1).sum()),"duplicate_folders":du.folder_patient.unique().tolist(),
"patientid_to_multiple_folders":int((d.groupby(["center","PatientID"]).folder_patient.nunique()>1).sum()),
"folder_to_multiple_patientids":int((d.groupby(["center","folder_patient"]).PatientID.nunique()>1).sum()),
"study_uid_cross_patient":int((d.groupby("StudyInstanceUID").folder_patient.nunique()>1).sum()),
"series_uid_cross_patient":int((d.groupby("SeriesInstanceUID").folder_patient.nunique()>1).sum()),
"study_date_missing":int(m.study_date.isna().sum()),"internal_study_patient_rows":len(m),"image_after_surgery_rows":int(m.days_before_surgery.lt(0).sum()),"same_day_rows":int(m.days_before_surgery.eq(0).sum()),"days_before_min":float(m.days_before_surgery.min()),"days_before_max":float(m.days_before_surgery.max()),"days_before_median":float(m.days_before_surgery.median()),
"centers_date_range":d.groupby("center").StudyDate.agg(["min","max"]).to_dict("index"),
"series_descriptions":d.groupby(["center","folder_series"]).SeriesDescription.nunique().to_dict().__str__()}
# Store protocol counts by patient/series, not image-count-weighted.
proto=d.drop_duplicates(["center","folder_patient","folder_series","Manufacturer","MagneticFieldStrength","SeriesDescription"])
proto.groupby(["center","folder_series","Manufacturer","MagneticFieldStrength","SeriesDescription"]).size().rename("patient_series_n").reset_index().to_csv(O/"scanner_protocol_summary.csv",index=False)
(O/"image_supplement.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(r,ensure_ascii=True))


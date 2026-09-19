"""Explicit versioned research entry point. Default: read-only status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
STAGES={
 'v2':{'state':'selective_risk_v2/run_all.json','run':['v2_pipeline.py','synthetic_boundary_v2.py','validate_v2.py','v2_robustness.py','report_v2.py'],'report':['report_v2.py'],'clinical':True},
 'v3':{'state':'boundary_v3/run.json','run':['boundary_v3.py'],'report':['boundary_v3.py']},
 'v4':{'state':'locked_real_v4/run.json','run':['locked_real_v4.py'],'report':[],'clinical':True},
 'v5':{'state':'mechanism_training_v5/formal/run.json','run':['mechanism_training_v5.py','report_mechanism_v5.py'],'report':['report_mechanism_v5.py'],'smoke':['mechanism_training_v5.py','--smoke']},
 'v6':{'state':'fixed_sequence_v6/run.json','run':['run_fixed_sequence_v6.py','report_fixed_sequence_v6.py'],'report':['report_fixed_sequence_v6.py'],'clinical':True},
 'v7':{'state':'calibration_information_v7/formal/run.json','run':['calibration_information_v7.py','report_calibration_v7.py'],'report':['report_calibration_v7.py'],'smoke':['calibration_information_v7.py','--smoke']},
}
def state(stage):
 path=ROOT/'results'/STAGES[stage]['state']
 return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'NOT_RUN'}
def clinical_preflight():
 path=ROOT/'results/selective_risk_v2/data_freeze_manifest.json'
 manifest=json.loads(path.read_text(encoding='utf-8'))
 assert manifest['technical_freeze']=='PASS','E0 technical audit has not passed'
 for source,expected in [(Path(manifest['source']),manifest['source_sha256']),(ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv',manifest['derived_data_sha256'])]:
  actual=hashlib.sha256(source.read_bytes()).hexdigest()
  if actual!=expected:raise RuntimeError('Input changed since audited version: '+str(source))
 print('E0 hashes match. Clinical adjudication remains partial; exploratory analyses only.',flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=STAGES)
 p.add_argument('--mode',choices=['status','run','report','smoke'],default='status')
 p.add_argument('--dry-run',action='store_true',help='Print commands without executing or changing results')
 p.add_argument('--sync-github',action='store_true',help='After success, push only the approved code/docs allowlist')
 args=p.parse_args()
 if args.mode=='status':
  print(json.dumps({s:state(s) for s in ([args.stage] if args.stage else STAGES)},ensure_ascii=False,indent=2));return
 if not args.stage:p.error('--stage is required for execution')
 spec=STAGES[args.stage]
 if args.mode=='smoke':
  if 'smoke' not in spec:p.error('A separate smoke mode exists only for v5 and v7')
  entry=spec['smoke'];commands=[[sys.executable,str(ROOT/'src'/entry[0]),*entry[1:]]]
 else:
  scripts=spec[args.mode]
  if not scripts:p.error('This version has no report-only entry; use the saved report or explicitly rerun it')
  commands=[[sys.executable,str(ROOT/'src'/name)] for name in scripts]
 for command in commands:print(subprocess.list2cmdline(command),flush=True)
 if args.dry_run:return
 status=state(args.stage)['status']
 if args.mode in ['run','report'] and status=='RUNNING':
  raise RuntimeError('Stage marked RUNNING. Inspect its process/state before restarting; duplicate execution is blocked.')
 if args.mode=='report' and status!='COMPLETED':raise RuntimeError('A completed run is required before reporting')
 if args.mode=='run' and spec.get('clinical'):clinical_preflight()
 lock=ROOT/'results'/(args.stage+'.entry.lock')
 lock.parent.mkdir(exist_ok=True)
 try:
  with lock.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'stage':args.stage,'mode':args.mode},stream)
 except FileExistsError as exc:
  raise RuntimeError('Another versioned entry owns '+str(lock)+'. Inspect its process before recovery.') from exc
 try:
  for command in commands:subprocess.run(command,cwd=ROOT,check=True)
 finally:
  lock.unlink()
 if args.sync_github:
  subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'sync_github.ps1'),'-Message',f'Run {args.stage} {args.mode}'],cwd=ROOT,check=True)
if __name__=='__main__':main()

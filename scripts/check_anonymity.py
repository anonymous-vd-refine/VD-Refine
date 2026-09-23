"""Check release text, archive extras and trusted inference-checkpoint metadata."""
import argparse,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    findings=[];checked=0
    patterns=[re.compile(r'/(?:home|Users|mnt)/[^\s\"\'<>]+'),re.compile(r'\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)\d{1,3}\.\d{1,3}\b')]
    for f in ROOT.rglob('*'):
        if not f.is_file():continue
        rel=f.relative_to(ROOT)
        if any(p in ('.venv','data','runs','__pycache__','.git') for p in rel.parts):continue
        if f.suffix in ('.py','.md','.json','.toml','.txt','.csv'):
            checked+=1
            for i,line in enumerate(f.read_text().splitlines(),1):
                for pat in patterns:
                    if pat.search(line) and not (f.name=='requirements-lock.txt' and '==' in line):findings.append({'file':str(rel),'line':i,'kind':'path or private address'})
        if f.suffix=='.pth':
            import torch
            ck=torch.load(f,map_location='cpu',weights_only=True)
            if set(ck)!={'network_weights','trainer_name','init_args','inference_allowed_mirroring_axes'}:findings.append({'file':str(rel),'kind':'unexpected checkpoint fields'})
            meta=json.dumps({k:v for k,v in ck.items() if k!='network_weights'})
            for pat in patterns:
                if pat.search(meta):findings.append({'file':str(rel),'kind':'checkpoint identity path'})
    report={'checked_text_files':checked,'findings':findings,'scope':'Absolute personal paths, private IP patterns and checkpoint field allowlist; third-party attribution intentionally retained. This does not guarantee anonymity against every external correlation.'}
    if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if findings:raise SystemExit(1)
if __name__=='__main__':main()

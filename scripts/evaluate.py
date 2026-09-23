"""Strict full-cohort BraTS Dice; finite depths and analytic inf share one evaluator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {'WT': (1, 2, 3), 'TC': (2, 3), 'ET': (3,)}
STEPS = ('0', '1', '2', '4', '8', '16', '32', 'inf')

def dice(gt, pred):
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    return float('nan') if tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn))

def mean(values):
    v = np.asarray(values, dtype=float)
    return float(np.nanmean(v)) if np.any(~np.isnan(v)) else float('nan')

def case_set(folder, expected):
    found = {p.name[:-7] for p in Path(folder).glob('*.nii.gz')}
    if found != set(expected):
        raise ValueError(f'Case set mismatch in {folder}: missing={sorted(set(expected)-found)}, extra={sorted(found-set(expected))}')

def evaluate_folder(gt_dir, pred_dir, expected):
    case_set(gt_dir, expected); case_set(pred_dir, expected)
    rows = []
    for case in expected:
        g = sitk.ReadImage(str(Path(gt_dir)/(case+'.nii.gz')))
        p = sitk.ReadImage(str(Path(pred_dir)/(case+'.nii.gz')))
        if g.GetDimension()!=3 or p.GetDimension()!=3 or g.GetSize()!=p.GetSize():
            raise ValueError(f'{case}: dimension/shape mismatch')
        for attr in ('GetSpacing', 'GetOrigin', 'GetDirection'):
            if not np.allclose(getattr(g,attr)(),getattr(p,attr)(),rtol=0,atol=1e-5):
                raise ValueError(f'{case}: physical geometry mismatch: {attr}')
        ga,pa=sitk.GetArrayFromImage(g),sitk.GetArrayFromImage(p)
        if not np.isin(ga,(0,1,2,3)).all() or not np.isin(pa,(0,1,2,3)).all():
            raise ValueError(f'{case}: expected converted labels 0/1/2/3')
        row={'case':case}
        for name,labels in REGIONS.items():row[name+'_Dice']=dice(np.isin(ga,labels),np.isin(pa,labels))
        row['Avg_Dice']=mean([row[n+'_Dice'] for n in REGIONS]);rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path,help='Contains testK0_best ... testKinf_best')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cohort',type=Path,default=ROOT/'dataset/cohort.json')
    p.add_argument('--steps',nargs='+',choices=STEPS,default=list(STEPS))
    args=p.parse_args()
    cohort=json.loads(args.cohort.read_text()); expected=cohort['test']
    if len(expected)!=74 or len(set(expected))!=74:raise ValueError('Expected exactly 74 unique test cases')
    if len(set(args.steps))!=len(args.steps):raise ValueError('Duplicate modes')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output directory is not empty; use a new output directory')
    rows=[];summary=[]
    for k in args.steps:
        kr=evaluate_folder(args.gt,args.predictions/f'testK{k}_best',expected)
        rows.extend({'K':k,**r} for r in kr)
        rec={'K':k,'n_cases':len(kr)}
        for name in (*REGIONS,'Avg'):
            vals=[r[name+'_Dice'] for r in kr]
            rec[name+'_Dice_valid_mean']=mean(vals)
            rec[name+'_Dice_nan_count']=sum(np.isnan(v) for v in vals)
        summary.append(rec)
        print(f"K={k}, n={len(kr)}, Avg Dice={rec['Avg_Dice_valid_mean']:.10f}")
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in [('scaling_dice.csv',summary),('scaling_dice_cases.csv',rows)]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (args.output/'evaluation.json').write_text(json.dumps({'n_cases':74,'steps':args.steps,'cohort_sha256':hashlib.sha256(args.cohort.read_bytes()).hexdigest(),'aggregation':'per-case nanmean(WT,TC,ET), then nanmean across cases','note':'Checkpoint and TTA provenance must come from the prediction runner metadata.'},indent=2)+'\n')
if __name__=='__main__':main()

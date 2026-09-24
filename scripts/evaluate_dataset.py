"""Strict full-cohort LiTS and DRIVE Dice; finite depths and analytic inf share one evaluator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {}
DATASET = ''
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
    ext='.nii.gz' if DATASET=='lits' else '.tif'
    for folder in (gt_dir,pred_dir):
        found={p.name[:-len(ext)] for p in Path(folder).glob('*'+ext)}
        if found!=set(expected):raise ValueError(f'Case set mismatch in {folder}')
    rows = []
    for case in expected:
        if DATASET=='lits':
            g=sitk.ReadImage(str(Path(gt_dir)/(case+ext)));p=sitk.ReadImage(str(Path(pred_dir)/(case+ext)))
            if g.GetDimension()!=3 or p.GetDimension()!=3 or g.GetSize()!=p.GetSize():raise ValueError(f'{case}: shape mismatch')
            for attr in ('GetSpacing','GetOrigin','GetDirection'):
                if not np.allclose(getattr(g,attr)(),getattr(p,attr)(),rtol=0,atol=1e-5):raise ValueError(f'{case}: geometry mismatch')
            ga,pa=sitk.GetArrayFromImage(g),sitk.GetArrayFromImage(p)
        else:
            import tifffile
            ga=tifffile.imread(Path(gt_dir)/(case+ext));pa=tifffile.imread(Path(pred_dir)/(case+ext))
            if ga.ndim!=2 or ga.shape!=pa.shape:raise ValueError(f'{case}: shape mismatch')
        allowed=(0,1,2) if DATASET=='lits' else (0,1)
        if not np.isin(ga,allowed).all() or not np.isin(pa,allowed).all():raise ValueError(f'{case}: unexpected labels')
        row={'case':case}
        for name,labels in REGIONS.items():row[name+'_Dice']=dice(np.isin(ga,labels),np.isin(pa,labels))
        row['Avg_Dice']=mean([row[n+'_Dice'] for n in REGIONS]);rows.append(row)
    return rows

def main():
    global REGIONS,DATASET
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['lits','drive'],required=True)
    p.add_argument('--gt',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path,help='Contains testK0_best ... testKinf_best')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cohort',type=Path,default=None)
    p.add_argument('--steps',nargs='+',choices=STEPS,default=list(STEPS))
    args=p.parse_args()
    DATASET=args.dataset;REGIONS={'Liver':(1,2),'Tumor':(2,)} if DATASET=='lits' else {'Vessel':(1,)}
    count=26 if DATASET=='lits' else 20;tag='final' if DATASET=='lits' else 'best'
    args.cohort=args.cohort or ROOT/'dataset'/DATASET/'cohort.json'
    cohort=json.loads(args.cohort.read_text()); expected=cohort['test']
    if len(expected)!=count or len(set(expected))!=count:raise ValueError(f'Expected {count} unique cases')
    if len(set(args.steps))!=len(args.steps):raise ValueError('Duplicate modes')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output directory is not empty; use a new output directory')
    rows=[];summary=[]
    for k in args.steps:
        kr=evaluate_folder(args.gt,args.predictions/f'testK{k}_{tag}',expected)
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
    (args.output/'evaluation.json').write_text(json.dumps({'n_cases':count,'steps':args.steps,'cohort_sha256':hashlib.sha256(args.cohort.read_bytes()).hexdigest(),'aggregation':'per-case nanmean of region Dice, then nanmean across cases','note':'Checkpoint and TTA provenance must come from the prediction runner metadata.'},indent=2)+'\n')
if __name__=='__main__':main()

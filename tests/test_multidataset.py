"""Synthetic checks for cohort completeness, region semantics, geometry and converters."""
import json,tempfile,sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import tifffile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_dataset as ev
import prepare_dataset as prep
assert np.isnan(ev.dice(np.zeros(3,bool),np.zeros(3,bool)))
assert ev.dice(np.zeros(3,bool),np.ones(3,bool))==0
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for ds in ['lits','drive']:
  ev.DATASET=ds;ev.REGIONS={'Liver':(1,2),'Tumor':(2,)} if ds=='lits' else {'Vessel':(1,)}
  cohort=json.loads((ROOT/'dataset'/ds/'cohort.json').read_text());src=root/(ds+'-source');raw=root/(ds+'-raw');pred=root/(ds+'-pred');pred.mkdir()
  for case in cohort['train']+cohort['val']+cohort['test']:
   n=int(case.split('_')[-1]);a=np.zeros((3,4,5) if ds=='lits' else (4,5),np.uint8);a.flat[0]=1;a.flat[1]=2 if ds=='lits' else 1
   if ds=='lits':
    for sub in ['imagesTr','labelsTr']:
     (src/sub).mkdir(exist_ok=True,parents=True);sitk.WriteImage(sitk.GetImageFromArray(a),str(src/sub/f'liver_{n-1}.nii.gz'))
   else:
    split='training' if n>=21 else 'test'
    for sub in ['images','1st_manual']:(src/split/sub).mkdir(exist_ok=True,parents=True)
    Image.fromarray(np.zeros((4,5,3),np.uint8)).save(src/split/'images'/f'{n:02d}_{split}.tif');Image.fromarray(a*255).save(src/split/'1st_manual'/f'{n:02d}_manual1.gif')
  prep.convert(ds,src,raw)
  import shutil
  for f in (raw/'labelsTs').iterdir():shutil.copyfile(f,pred/f.name)
  rows=ev.evaluate_folder(raw/'labelsTs',pred,cohort['test']);assert all(r['Avg_Dice']==1 for r in rows)
  target=next(pred.iterdir());target.unlink()
  try:ev.evaluate_folder(raw/'labelsTs',pred,cohort['test'])
  except ValueError:pass
  else:raise AssertionError('Missing test case must fail')
 # Whole-liver union includes tumor, while tumor Dice remains separate.
 assert ev.dice(np.isin([1,2],[1,2]),np.isin([2,1],[1,2]))==1
print('PASS LiTS/DRIVE conversion, exact cohorts, perfect masks, empty-mask semantics, missing-case rejection')

"""Convert legally obtained BraTS2020 training cases using the published fixed cohort."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
ROOT=Path(__file__).resolve().parents[1]

def convert_labels(arr):
    if not np.isin(arr,(0,1,2,4)).all():raise ValueError('Expected original BraTS labels {0,1,2,4}')
    # Original 1=necrotic/non-enhancing core, 2=edema, 4=enhancing tumor.
    out=np.zeros(arr.shape,dtype=np.uint8)
    out[arr==2]=1;out[arr==1]=2;out[arr==4]=3
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True,help='Directory containing BraTS20_Training_* case folders');p.add_argument('--output',type=Path,required=True,help='New Dataset802_BraTS directory');a=p.parse_args()
    cohort=json.loads((ROOT/'dataset/cohort.json').read_text())
    ids=cohort['train']+cohort['val']+cohort['test']
    if len(ids)!=369 or len(set(ids))!=369:raise ValueError('Invalid cohort')
    if a.output.exists():raise FileExistsError('Refusing to overwrite an existing dataset')
    # Preflight all required inputs before creating output.
    for case in ids:
        for suffix in ('t1','t1ce','t2','flair','seg'):
            f=a.source/case/f'{case}_{suffix}.nii.gz'
            if not f.is_file():raise FileNotFoundError(f)
    for d in ('imagesTr','labelsTr','imagesTs','labelsTs'):(a.output/d).mkdir(parents=True,exist_ok=True)
    for case in ids:
        split='Ts' if case in cohort['test'] else 'Tr'
        for ch,suffix in enumerate(('t1','t1ce','t2','flair')):
            shutil.copyfile(a.source/case/f'{case}_{suffix}.nii.gz',a.output/f'images{split}'/f'{case}_{ch:04d}.nii.gz')
        ref=sitk.ReadImage(str(a.source/case/f'{case}_seg.nii.gz'))
        out=sitk.GetImageFromArray(convert_labels(sitk.GetArrayFromImage(ref)));out.CopyInformation(ref)
        sitk.WriteImage(out,str(a.output/f'labels{split}'/f'{case}.nii.gz'),True)
    shutil.copyfile(ROOT/'dataset/dataset.json',a.output/'dataset.json')
    print('Converted 295 development (236 train + 59 validation) and 74 held-out test cases.')
if __name__=='__main__':main()

"""Convert provider-owned data to the fixed published LiTS or DRIVE cohort; validate filenames."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def validate(ds,raw):
    meta=ROOT/'dataset'/ds;d=json.loads((meta/'dataset.json').read_text());c=json.loads((meta/'cohort.json').read_text());ext=d['file_ending']
    groups=[set(c[k]) for k in ('train','val','test')]
    sizes=(92,13,26) if ds=='lits' else (16,4,20)
    if tuple(map(len,groups))!=sizes or any(groups[i]&groups[j] for i,j in [(0,1),(0,2),(1,2)]):raise ValueError('Invalid cohort')
    for suffix,ids in [('Tr',groups[0]|groups[1]),('Ts',groups[2])]:
        for folder,expected in [(f'labels{suffix}',{v+ext for v in ids}),(f'images{suffix}',{v+'_0000'+ext for v in ids})]:
            found={p.name for p in (raw/folder).glob('*'+ext)}
            if found!=expected:raise ValueError(f'{folder}: missing={sorted(expected-found)}, extra={sorted(found-expected)}')
    actual=json.loads((raw/'dataset.json').read_text())
    for key in ('channel_names','labels','numTraining','file_ending'):
        if actual[key]!=d[key]:raise ValueError(f'dataset.json mismatch: {key}')
    print(f'PASS {ds}: {sizes}; disjoint cohort and exact image/label filenames')

def convert(ds,source,raw):
    import numpy as np
    from PIL import Image
    import tifffile
    meta=ROOT/'dataset'/ds;c=json.loads((meta/'cohort.json').read_text())
    if raw.exists():raise FileExistsError('Output already exists; choose a new directory')
    jobs=[]
    for suffix,ids in [('Tr',c['train']+c['val']),('Ts',c['test'])]:
        for case in ids:
            n=int(case.split('_')[-1])
            if ds=='lits':
                # Verified MSD Task03_Liver mapping: liver_n -> Liver_{n+1:04d}.
                img=source/'imagesTr'/f'liver_{n-1}.nii.gz';lab=source/'labelsTr'/f'liver_{n-1}.nii.gz'
            else:
                split='training' if n>=21 else 'test';kind='training' if n>=21 else 'test'
                img=source/split/'images'/f'{n:02d}_{kind}.tif';lab=source/split/'1st_manual'/f'{n:02d}_manual1.gif'
            if not img.is_file() or not lab.is_file():raise FileNotFoundError(f'Missing source pair: {img}, {lab}')
            jobs.append((suffix,case,img,lab))
    for sub in ('imagesTr','labelsTr','imagesTs','labelsTs'):(raw/sub).mkdir(parents=True)
    for suffix,case,img,lab in jobs:
        if ds=='lits':
            shutil.copyfile(img,raw/f'images{suffix}'/(case+'_0000.nii.gz'));shutil.copyfile(lab,raw/f'labels{suffix}'/(case+'.nii.gz'))
        else:
            rgb=np.asarray(Image.open(img).convert('RGB'));mask=np.asarray(Image.open(lab))
            if mask.ndim!=2 or rgb.shape[:2]!=mask.shape:raise ValueError(f'{case}: source shape mismatch')
            tifffile.imwrite(raw/f'images{suffix}'/(case+'_0000.tif'),rgb)
            tifffile.imwrite(raw/f'labels{suffix}'/(case+'.tif'),(mask>0).astype('uint8'))
    shutil.copyfile(meta/'dataset.json',raw/'dataset.json');validate(ds,raw)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['convert','validate']);p.add_argument('--dataset',choices=['lits','drive'],required=True);p.add_argument('--source',type=Path);p.add_argument('--raw',required=True,type=Path);a=p.parse_args()
    if a.action=='convert':
        if a.source is None:p.error('--source is required for convert')
        convert(a.dataset,a.source,a.raw)
    else:validate(a.dataset,a.raw)

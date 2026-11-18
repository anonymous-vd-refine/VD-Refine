"""Validate the exact, disjoint BraTS cohort and four modality filenames."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def validate(raw):
    c=json.loads((ROOT/'dataset/cohort.json').read_text())
    tr,va,te=(set(c[k]) for k in ('train','val','test'))
    if (len(tr),len(va),len(te))!=(236,59,74) or tr&va or tr&te or va&te:raise ValueError('Invalid split')
    for suffix,ids in [('Tr',tr|va),('Ts',te)]:
        labels={f.name for f in (raw/f'labels{suffix}').glob('*.nii.gz')}
        images={f.name for f in (raw/f'images{suffix}').glob('*.nii.gz')}
        if labels!={f'{c}.nii.gz' for c in ids}:raise ValueError(f'labels{suffix}: cohort mismatch')
        if images!={f'{c}_{ch:04d}.nii.gz' for c in ids for ch in range(4)}:raise ValueError(f'images{suffix}: cohort/modality mismatch')
    d=json.loads((raw/'dataset.json').read_text());ref=json.loads((ROOT/'dataset/dataset.json').read_text())
    for key in ('channel_names','labels','regions_class_order','numTraining','file_ending'):
        if d[key]!=ref[key]:raise ValueError(f'dataset.json: {key} mismatch')
    print('PASS: 236 train / 59 validation / 74 test, no case overlap, four modalities.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);validate(p.parse_args().raw)

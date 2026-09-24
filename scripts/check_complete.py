"""Do not report an unfinished run as a completed E100 experiment."""
import argparse
from pathlib import Path
import torch
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    ck=torch.load(a.model/'fold_0/checkpoint_final.pth',map_location='cpu',weights_only=False)
    n=len(ck['logging']['train_losses'])
    if n!=100:raise ValueError(f'Expected 100 logged training epochs, found {n}')
    if not (a.model/'fold_0/checkpoint_best.pth').is_file():raise FileNotFoundError('Missing checkpoint_best.pth')
    print('PASS: E100 final checkpoint and best checkpoint exist.')

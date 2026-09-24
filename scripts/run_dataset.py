"""Portable LiTS and DRIVE runner. Never overwrites training, preprocessing, or prediction outputs."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preprocess','train','resume','predict','evaluate'])
    p.add_argument('--dataset',choices=['lits','drive'],required=True)
    p.add_argument('--model',type=Path,help='External historical model folder with checkpoint, plans.json and dataset.json')
    p.add_argument('--data',type=Path,default=ROOT/'data',help='Contains nnUNet_raw, nnUNet_preprocessed, nnUNet_results')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--gpu',default='0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--steps',nargs='+',choices=['0','1','2','4','8','16','32','inf'],default=['0','1','2','4','8','16','32','inf'])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();a.data=a.data.resolve();a.output=a.output.resolve()
    if a.model and a.action not in ('predict','evaluate'):p.error('--model is only for prediction/evaluation')
    if a.workers<1:p.error('--workers must be positive')
    cfg=json.loads((ROOT/f'configs/{a.dataset}.json').read_text());ds=cfg['dataset'];configuration=cfg['configuration'];meta=ROOT/'dataset'/a.dataset;tag=cfg['checkpoint'].removeprefix('checkpoint_').removesuffix('.pth')
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('LITE_','NNUNET_')) or k in ('VIRTUALDEEP_WARM_START_A','FILM_WARM_START_A','CONTINUE','PRETRAINED_WEIGHTS','INPUT_DIR'):
            env.pop(k,None)
    env.update(PYTHONPATH=str(ROOT),PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=a.gpu,nnUNet_n_proc_DA=str(a.workers),nnUNet_compile='false',NNUNET_TIFF_ZLIB='1')
    for k in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'):env[k]=str(a.data/k)
    raw=a.data/'nnUNet_raw'/ds;pre=a.data/'nnUNet_preprocessed'/ds
    model=a.data/'nnUNet_results'/ds/(cfg['trainer']+'__nnUNetPlans__'+configuration)
    run=a.output/a.dataset/('external' if a.model else 'paper')
    if a.model:model=a.model.resolve()
    def call(args):
        print(' '.join(map(str,args)),flush=True)
        if not a.dry_run:subprocess.run(list(map(str,args)),env=env,cwd=ROOT,check=True)
    if a.action=='preprocess':
        if (pre/('nnUNetPlans_'+configuration)).exists():raise FileExistsError('Preprocessed data already exists')
        call([sys.executable,ROOT/'scripts/prepare_dataset.py','validate','--dataset',a.dataset,'--raw',raw])
        if not a.dry_run:
            pre.mkdir(parents=True,exist_ok=True)
            for name in ('dataset.json','nnUNetPlans.json','dataset_fingerprint.json','splits_final.json'):
                target=pre/name;source=meta/name
                if target.exists() and target.read_bytes()!=source.read_bytes():raise FileExistsError(f'Different metadata exists: {target}')
                shutil.copyfile(source,target)
        call([sys.executable,'-c',f"from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor; DefaultPreprocessor().run('{ds}','{configuration}','nnUNetPlans',{a.workers})"])
    elif a.action in ('train','resume'):
        if a.action=='train' and model.exists():raise FileExistsError('Training output exists: choose resume or a fresh --data root')
        if not a.dry_run:
            for name in ('splits_final.json','nnUNetPlans.json','dataset.json'):
                saved=json.loads((pre/name).read_text())
                if saved!=json.loads((meta/name).read_text()):raise ValueError(f'{name} differs from published metadata')
        args=[sys.executable,'-m','nnunetv2.run.run_training',ds,configuration,'0','-tr',cfg['trainer'],'-p','nnUNetPlans','-num_gpus','1']
        if a.action=='resume':
            if not any((model/'fold_0'/n).is_file() for n in ('checkpoint_latest.pth','checkpoint_best.pth')):raise FileNotFoundError('No resumable checkpoint')
            if (model/'fold_0/checkpoint_final.pth').exists():raise RuntimeError('Training already complete')
            args+=['--c']
        call(args)
    elif a.action=='predict':
        checkpoint=model/'fold_0'/cfg['checkpoint']
        if not a.dry_run:
            call([sys.executable,ROOT/'scripts/prepare_dataset.py','validate','--dataset',a.dataset,'--raw',raw])
            if not a.model:call([sys.executable,ROOT/'scripts/check_complete.py','--model',model])
            if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        for k in a.steps:
            out=run/f'testK{k}_{tag}'
            if out.exists():raise FileExistsError(f'Prediction output exists: {out}; choose a new --output')
            env['LITE_REFINER_INF_EXTRAPOLATE']='1' if k=='inf' else '0'
            env.pop('LITE_REFINER_TEST_STEPS',None)
            if k!='inf':env['LITE_REFINER_TEST_STEPS']=k
            call([sys.executable,'-c','from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder; predict_entry_point_modelfolder()', '-i',raw/'imagesTs','-o',out,'-m',model,'-f','0','-chk',cfg['checkpoint'],'-step_size','0.5','-npp','2','-nps','2',*(['--disable_tta'] if not cfg['tta'] else [])])
            if not a.dry_run:
                metadata={'version':'1.1.0','mode':k,'checkpoint':cfg['checkpoint'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'tta':cfg['tta'],'fold':0,'configuration':configuration,'cohort_sha256':hashlib.sha256((meta/'cohort.json').read_bytes()).hexdigest()}
                (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    else:
        call([sys.executable,ROOT/'scripts/evaluate_dataset.py','--dataset',a.dataset,'--gt',raw/'labelsTs','--predictions',run,'--output',run/'evaluation','--steps',*a.steps])
if __name__=='__main__':main()

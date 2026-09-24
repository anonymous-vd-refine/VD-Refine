"""CPU equivalence probe: source/release outputs, fast decode, and stage updates.
Run separately with each package root on PYTHONPATH, then compare the JSON reports.
Uses a deterministic synthetic 64^3 patch; does not replace full-volume evaluation.
"""
import argparse,hashlib,importlib,json,os,random,tempfile
from pathlib import Path
import numpy as np
import torch

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def state_digest(net):
    h=hashlib.sha256()
    for k,t in sorted(net.state_dict().items()):h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def reset_rng():random.seed(0);np.random.seed(0);torch.manual_seed(0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-package',type=Path,required=True);a=p.parse_args()
    for key in list(os.environ):
        if key.startswith('LITE_') or key in ('VIRTUALDEEP_WARM_START_A','FILM_WARM_START_A'):os.environ.pop(key)
    with tempfile.TemporaryDirectory() as td:
        for k in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'):os.environ[k]=str(Path(td)/k)
        os.environ['nnUNet_compile']='false'
        torch.set_num_threads(4);reset_rng()
        import nnunetv2
        assert Path(nnunetv2.__file__).resolve().parent==a.expected_package.resolve()
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        plans=json.loads((a.assets/'dataset/nnUNetPlans.json').read_text());ds=json.loads((a.assets/'dataset/dataset.json').read_text())
        name='nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2'
        cls=getattr(importlib.import_module('nnunetv2.training.nnUNetTrainer.'+name),name)
        obj=cls(plans,'3d_fullres',0,ds,device=torch.device('cpu'));assert obj.num_epochs==100
        obj.initialize();net=obj.network
        assert not obj.enable_deep_supervision
        assert sum(p.numel() for p in net.parameters())==4399737
        assert [len(s.refine) for s in net.proposal_stages]==[1]*5
        assert type(obj.optimizer).__name__=='AdamW'
        assert obj.initial_lr==1e-4 and obj.weight_decay==1e-5 and not obj._train_film_in_b
        ck=torch.load(a.assets/'weights/historical/fold_0/checkpoint_best.pth',map_location='cpu',weights_only=True)
        net.load_state_dict(ck['network_weights'],strict=True);net.eval()
        reset_rng();x=torch.randn(1,4,64,64,64)
        report={'parameters':4399737,'decoder_blocks':[1]*5,'epochs':100,'optimizer':'AdamW','input_shape':list(x.shape),'device':'cpu','dtype':'float32','inference':{},'training':{}}
        counts={}
        def hook(name):
            def f(*args):counts[name]=counts.get(name,0)+1
            return f
        handles=[net.mask_head.register_forward_hook(hook('head')),net.single_refiner.shared_block.register_forward_hook(hook('refiner'))]
        handles += [s.register_forward_hook(hook(f'tail{i}')) for i,s in enumerate(net.proposal_stages) if i>net.single_refine_index]
        for k in (0,1,2,4,8,16,32):
            counts.clear()
            with torch.inference_mode():fast=net(x,recurrent_steps=k).clone()
            fast_counts=dict(counts)
            assert fast_counts['head']==1 and fast_counts.get('refiner',0)==k
            assert all(v==1 for n,v in fast_counts.items() if n.startswith('tail'))
            with torch.inference_mode():slow=net(x,recurrent_steps=k,decode_all_steps=True)
            assert torch.equal(fast,slow),f'fast/legacy K={k}'
            assert torch.isfinite(fast).all()
            report['inference'][str(k)]={'sha256':digest(fast),'fast_equals_legacy':True,'counts':fast_counts}
            print('PASS inference',k,flush=True)
        counts.clear();os.environ['LITE_REFINER_INF_EXTRAPOLATE']='1'
        with torch.inference_mode():out=net(x)
        assert torch.isfinite(out).all() and counts['head']==1 and counts['refiner']==8
        report['inference']['inf']={'sha256':digest(out),'counts':dict(counts)}
        os.environ.pop('LITE_REFINER_INF_EXTRAPOLATE')
        for h in handles:h.remove()
        # Predictor loading must resolve the packaged historical trainer and exact weights.
        predictor=nnUNetPredictor(device=torch.device('cpu'),perform_everything_on_device=False)
        predictor.initialize_from_trained_model_folder(str(a.assets/'weights/historical'),(0,),'checkpoint_best.pth')
        predictor.network.load_state_dict(predictor.list_of_parameters[0],strict=True)
        assert state_digest(predictor.network)==state_digest(net)
        del predictor
        target=torch.zeros(1,3,64,64,64);target[:,:,16:48,16:48,16:48]=1
        for epoch in (0,50,70,80,90):
            reset_rng();net.load_state_dict(ck['network_weights']);net.train();obj.current_epoch=epoch
            obj.optimizer,obj.lr_scheduler=obj.configure_optimizers();obj.lr_scheduler.step(epoch)
            stage=obj._stage_spec_at(epoch)[0];obj._set_stage_trainability(stage)
            active=[n for n,p in net.named_parameters() if p.requires_grad]
            if stage.startswith('B'):assert len(active)==11 and all(n.startswith('single_refiner.shared_block.') for n in active)
            result=obj.train_step({'data':x,'target':target})
            assert all(np.isfinite(v).all() for v in result.values())
            if stage.startswith('B'):
                assert all(torch.equal(t,ck['network_weights'][n]) for n,t in net.state_dict().items() if not n.startswith('single_refiner.shared_block.'))
            report['training'][stage]={'losses':{k:float(v) for k,v in result.items()},'updated_state_sha256':state_digest(net),'trainable_tensors':len(active),'lr_groups':{g['name']:g['lr'] for g in obj.optimizer.param_groups}}
            print('PASS training',stage,flush=True)
        # Validate RNG schedule range without changing the actual recipe.
        for epoch,bounds in ((50,(10,16)),(80,(10,32))):
            draws=[obj._virtual_branch_schedule(epoch)[0] for _ in range(500)]
            assert min(draws)>=bounds[0] and max(draws)<=bounds[1]
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

"""Check that incomplete checkpoints cannot pass the E100 gate."""
import subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class CompletionTests(unittest.TestCase):
    def test_epoch_gate(self):
        with tempfile.TemporaryDirectory() as td:
            model=Path(td);fold=model/'fold_0';fold.mkdir();(fold/'checkpoint_best.pth').touch()
            for n,expected in ((99,False),(100,True),(101,False)):
                torch.save({'logging':{'train_losses':[0.0]*n}},fold/'checkpoint_final.pth')
                result=subprocess.run([sys.executable,str(ROOT/'scripts/check_complete.py'),'--model',str(model)],capture_output=True)
                self.assertEqual(result.returncode==0,expected)
    def test_missing_final(self):
        with tempfile.TemporaryDirectory() as td:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/check_complete.py'),'--model',td],capture_output=True)
            self.assertNotEqual(result.returncode,0)
if __name__=='__main__':unittest.main()

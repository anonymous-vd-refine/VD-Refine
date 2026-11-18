import importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
import SimpleITK as sitk
ROOT=Path(__file__).resolve().parents[1]
def module(name):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
E=module('evaluate');C=module('convert_brats')
class EvaluationTests(unittest.TestCase):
    def test_empty_region_semantics(self):
        z=np.zeros((3,3),bool);o=np.ones((3,3),bool)
        self.assertTrue(np.isnan(E.dice(z,z)));self.assertEqual(E.dice(z,o),0);self.assertEqual(E.dice(o,o),1)
    def test_label_conversion(self):
        np.testing.assert_array_equal(C.convert_labels(np.array([0,1,2,4])),[0,2,1,3])
        with self.assertRaises(ValueError):C.convert_labels(np.array([3]))
    def test_missing_extra_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'g';p=Path(td)/'p';g.mkdir();p.mkdir()
            a=sitk.GetImageFromArray(np.ones((4,4,4),np.uint8))
            sitk.WriteImage(a,str(g/'a.nii.gz'))
            with self.assertRaises(ValueError):E.evaluate_folder(g,p,['a'])
            sitk.WriteImage(a,str(p/'a.nii.gz'));self.assertEqual(E.evaluate_folder(g,p,['a'])[0]['WT_Dice'],1)
            sitk.WriteImage(a,str(p/'extra.nii.gz'))
            with self.assertRaises(ValueError):E.evaluate_folder(g,p,['a'])
            (p/'extra.nii.gz').unlink();a.SetSpacing((2,1,1));sitk.WriteImage(a,str(p/'a.nii.gz'))
            with self.assertRaises(ValueError):E.evaluate_folder(g,p,['a'])
    def test_per_case_aggregation(self):
        vals=[E.mean([1,1,float('nan')]),E.mean([0,0,1])]
        self.assertAlmostEqual(E.mean(vals),2/3)
if __name__=='__main__':unittest.main()

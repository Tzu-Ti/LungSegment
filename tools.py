__author__ = 'Titi Wei'
import torch
from torch.nn import functional as F
import numpy as np
import os

# visualize segmentation
def visual_label(x, num_classes):
    x = x / num_classes * 255
    x = x.type(torch.uint8)
    return x

def visual_seg(x, num_classes, is_gt):
    if not is_gt: # if not ground truth, that is prediction, need to do softmax
        x = F.softmax(x, dim=1)
    n = torch.argmax(x, dim=1, keepdim=True).type(torch.uint8)
    n = visual_label(n, num_classes)
    return n

# saving 3D CT and prediction
import nibabel as nib
from data import utils
from data.preprocess import reorientation
class Saving():
    def __init__(self, json_path):
        self.labelmap = utils.open_json(json_path)
        
    def update(self, pred, idx):
        if idx == 0:
            self.preds = pred
        else:
            self.preds = torch.cat([self.preds, pred], dim=0)

    def mapping_back(self, pred):
        new = pred.copy()
        for k in self.labelmap.keys():
            label = self.labelmap[k]
            if label == 0: continue
            loc = np.where(pred==int(label))
            if len(loc) == 1:
                continue
            else:
                X, Y, Z = loc
            for x, y, z in zip(X, Y, Z):
                new[x, y, z] = k
        return new

    def configure_seg(self, mapping_back=True):
        pred = self.preds.permute(1, 2, 0).cpu().numpy()
        if mapping_back:
            pred = self.mapping_back(pred)
        return pred
    
def save_niigz(ct, default_orient, name, saving_folder, **kwargs):
    orient_axcodes = tuple(default_orient)
    ct_axcodes = nib.aff2axcodes(ct.affine)
    ct_header = ct.header
    ct = ct.get_fdata()
    if ct_axcodes != orient_axcodes:
        print("Reorienting CT...")
        ct = reorientation(ct, ct_axcodes, orient_axcodes)
    
    ct = nib.Nifti1Image(ct, None, header=ct_header)
    preds = {}
    for k, v in kwargs.items():
        preds[k] = nib.Nifti1Image(v, None, header=ct_header)
    os.makedirs(saving_folder, exist_ok=True)
    ct.to_filename(os.path.join(saving_folder, '{}_CT.nii.gz'.format(name)))
    for k, v in preds.items():
        v.to_filename(os.path.join(saving_folder, '{}_{}.nii.gz'.format(name, k)))

# Metric
import pytorch_lightning as pl
from torchmetrics import MeanMetric

class MeasureMetric(pl.LightningModule):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        for m in metrics:
            setattr(self, m, MeanMetric().to('cuda'))

    def update(self, pred, target):
        if 'DICE' in self.metrics:
            d = dice(pred, target)
            self.DICE.update(d)

    def compute(self):
        results = {}
        for m in self.metrics:
            meanmetric = getattr(self, m)
            results[m] = meanmetric.compute()

        return results

def dice(pred, target):
    smooth = 1e-5
    B = pred.shape[0]
    m1 = pred.view(B, -1)
    m2 = target.view(B, -1)
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
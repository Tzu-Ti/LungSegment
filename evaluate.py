__author__ = 'Titi Wei'
from data.dataset import TestDataset, PredictDataset
import tools

from tqdm import tqdm
import argparse, yaml
import os, sys
import nibabel as nib

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from main import LitModel

def parser():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    # YAML path
    parser.add_argument('--yaml_path', required=True)
    # Inference, patient path
    parser.add_argument('--patient_path', required=True)
    parser.add_argument('--ct_path', required = '--predict' in sys.argv)
    parser.add_argument('--saving_folder', required='--predict' in sys.argv)

    return parser.parse_args()

class ModelFactory():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=cfg['model_name'],
            name='lightning_logs'
        )

        if args.test:
            Dataset = TestDataset(self.args.patient_path, lung_classes=cfg['lung_classes'], tumor_classes=cfg['tumor_classes'])
            self.loader = DataLoader(Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
        elif args.predict:
            Dataset = PredictDataset(self.args.patient_path)
            self.loader = DataLoader(Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

        self.LungModel = LitModel.load_from_checkpoint(cfg['lung_ckpt_path'])
        self.TumorModel = LitModel.load_from_checkpoint(cfg['tumor_ckpt_path'])

        self.LungMMs = [tools.MeasureMetric(metrics=['DICE']) for _ in range(cfg['lung_classes'])]
        self.TumorMMs = [tools.MeasureMetric(metrics=['DICE']) for _ in range(cfg['tumor_classes'])]

    def calc_metric(self, pred, gt, num_classes, part):
        pred = pred.squeeze(1)
        s = F.one_hot(pred, num_classes=num_classes).type(torch.float32)
        s = s.permute(0, 3, 1, 2)
        for b in range(s.shape[0]):
            for i in range(num_classes):
                if part == 'lung':
                    self.LungMMs[i].update(s[b, i], gt[b, i])
                elif part == 'tumor':
                    self.TumorMMs[i].update(s[b, i], gt[b, i])

    def test(self):
        for idx, batch in enumerate(tqdm(self.loader)):
            batch = [b.type(torch.cuda.FloatTensor) for b in batch]
            ct, lung, tumor = batch

            # Predict Lung
            lung_pred = self.LungModel(ct)
            lung_pos = torch.where(lung_pred >= 1, 1, 0)
            lung_ct = torch.where(lung_pos == 1, ct, 0)

            # Predict Tumor
            tumor_pred = self.TumorModel(lung_ct)
            print(ct.shape, lung_pred.shape, tumor_pred.shape)

            # Calculate Metric
            self.calc_metric(lung_pred, lung, self.cfg['lung_classes'], 'lung')
            self.calc_metric(tumor_pred, tumor, self.cfg['tumor_classes'], 'tumor')

            self.tb_logger.experiment.add_images('ct', ct, idx)
            self.tb_logger.experiment.add_images('test/lung_pred', tools.visual_label(lung_pred, self.cfg['lung_classes']), idx)
            self.tb_logger.experiment.add_images('test/lung_pos', tools.visual_label(lung_pos, self.cfg['lung_classes']), idx)
            self.tb_logger.experiment.add_images('test/lung_ct', lung_ct, idx)
            self.tb_logger.experiment.add_images('test/tumor_pred', tools.visual_label(tumor_pred, self.cfg['tumor_classes']), idx)
            self.tb_logger.experiment.add_images('GT/lung', tools.visual_seg(lung, self.cfg['lung_classes'], is_gt=True), idx)
            self.tb_logger.experiment.add_images('GT/tumor', tools.visual_seg(tumor, self.cfg['tumor_classes'], is_gt=True), idx)

        LungMs = [self.LungMMs[i].compute() for i in range(self.cfg['lung_classes'])]
        TumorMs = [self.TumorMMs[i].compute() for i in range(self.cfg['tumor_classes'])]
        for i in range(self.cfg['lung_classes']):
            if i == 0:
                continue
            print('Lung Class {:02d}: {:.3f}'.format(i, LungMs[i]['DICE']))
        for i in range(self.cfg['tumor_classes']):
            if i == 0:
                continue
            print('Tumor Class {:02d}: {:.3f}'.format(i, TumorMs[i]['DICE']))

    
    def predict(self):
        LungS = tools.Saving(self.cfg['lung_json_path'])
        TumorS = tools.Saving(self.cfg['lung_json_path']) # not use label json
        for idx, batch in enumerate(tqdm(self.loader)):
            ct = batch.type(torch.cuda.FloatTensor)

            # Predict Lung
            lung_pred = self.LungModel(ct)
            lung_pos = torch.where(lung_pred >= 1, 1, 0)
            lung_ct = torch.where(lung_pos == 1, ct, 0)

            # Predict Tumor
            tumor_pred = self.TumorModel(lung_ct)
            print(ct.shape, lung_pred.shape, tumor_pred.shape)

            # Saving
            LungS.update(lung_pred.squeeze(1), idx)
            TumorS.update(tumor_pred.squeeze(1), idx)

        print("Configuring all predictions to 3D segmentation...")
        LungPred = LungS.configure_seg()
        TumorPred = TumorS.configure_seg(mapping_back=False)

        ct = nib.load(self.args.ct_path)
        name = self.args.ct_path.split('/')[-2]
        print("Saving...")
        tools.save_niigz(ct, self.cfg['default_orient'], name, self.args.saving_folder, Lung=LungPred, Tumor=TumorPred)

    def fast_run(self):
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(model=self.model, train_dataloaders=self.loader)

def main():
    args = parser()
    with open(args.yaml_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    factory = ModelFactory(args, cfg)
    if args.test:
        factory.test()
    elif args.predict:
        factory.predict()

if __name__ == '__main__':
    main()
__author__ = 'Titi Wei'
from data.dataset import LungDataset, TumorDataset
from models.UNet import UNet
import tools

import argparse, yaml
import os, sys
import nibabel as nib

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

def parser():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # Segment Part
    parser.add_argument('--LungSegment', action='store_true')
    parser.add_argument('--TumorSegment', action='store_true')
    # Mode
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--fast_run', action='store_true')
    # YAML path
    parser.add_argument('--yaml_path', required=True)
    # checkpoint path
    parser.add_argument('--ckpt_path', required='--test' in sys.argv or '--predict' in sys.argv or '--resume' in sys.argv)
    # Inference, patient path
    parser.add_argument('--patient_path', required='--predict' in sys.argv)
    parser.add_argument('--ct_path', required='--predict' in sys.argv)
    parser.add_argument('--saving_folder', required='--predict' in sys.argv)

    return parser.parse_args()

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.UNet = UNet(n_channels=1, conv_dim=32, n_classes=cfg['num_classes'], down_times=cfg['down_times'])
        self.CE = nn.CrossEntropyLoss()

        self.MMs = [tools.MeasureMetric(metrics=['DICE']) for _ in range(cfg['num_classes'])]

    def forward(self, ct):
        output = self.UNet(ct)

        s = F.softmax(output, dim=1)
        s = torch.argmax(s, dim=1, keepdim=True).type(torch.long)

        return s

    def training_step(self, batch, batch_idx):
        ct, seg = batch
        output = self.UNet(ct)

        CE = self.CE(output, seg)
        loss = CE

        # Log
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        show_num = self.cfg['show_num']
        self.tmp = {
            'ct': ct[:show_num],
            'seg': tools.visual_seg(seg[:show_num], num_classes=self.cfg['num_classes'], is_gt=True),
            'output': tools.visual_seg(output[:show_num], num_classes=self.cfg['num_classes'], is_gt=False)
        }

        return loss
    
    def on_train_epoch_end(self):
        self.logger.experiment.add_images('Train/ct', self.tmp['ct'], self.current_epoch)
        self.logger.experiment.add_images('Train/seg', self.tmp['seg'], self.current_epoch)
        self.logger.experiment.add_images('Train/output', self.tmp['output'], self.current_epoch)

    def validation_step(self, batch, batch_idx):
        ct, seg = batch
        output = self.UNet(ct)

        show_num = self.cfg['show_num']
        self.tmp = {
            'ct': ct[:show_num],
            'seg': tools.visual_seg(seg[:show_num], num_classes=self.cfg['num_classes'], is_gt=True),
            'output': tools.visual_seg(output[:show_num], num_classes=self.cfg['num_classes'], is_gt=False)
        }

    def on_validation_epoch_end(self):
        self.logger.experiment.add_images('Val/ct', self.tmp['ct'], self.current_epoch)
        self.logger.experiment.add_images('Val/seg', self.tmp['seg'], self.current_epoch)
        self.logger.experiment.add_images('Val/output', self.tmp['output'], self.current_epoch)

    def test_step(self, batch, batch_idx):
        ct, seg = batch
        output = self.UNet(ct)

        # Metric
        s = F.softmax(output, dim=1)
        s = torch.argmax(s, dim=1).type(torch.long)
        s = F.one_hot(s, num_classes=self.cfg['num_classes']).type(torch.float32)
        s = s.permute(0, 3, 1, 2)
        for b in range(s.shape[0]):
            for i in range(self.cfg['num_classes']):
                self.MMs[i].update(s[b, i], seg[b, i])

    def on_test_end(self):
        Ms = [self.MMs[i].compute() for i in range(self.cfg['num_classes'])]
        for i in range(self.cfg['num_classes']):
            if i == 0:
                continue
            print('Class {:02d}: {:.3f}'.format(i, Ms[i]['DICE']))

    def predict_step(self, batch, batch_idx):
        ct = batch
        output = self.UNet(ct)

        s = F.softmax(output, dim=1)
        s = torch.argmax(s, dim=1).type(torch.long)

        return s

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.UNet.parameters(), lr=self.cfg['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]

class ModelFactory():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=cfg['model_name'],
            name='lightning_logs'
        )

        if args.LungSegment:
            if args.train or args.fast_run or args.resume:
                Dataset = LungDataset(folder=cfg['folder'], list_path=cfg['list_path'], num_classes=cfg['num_classes'], train=True)
                self.loader = DataLoader(Dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
                Dataset = LungDataset(folder=cfg['folder'], list_path=cfg['val_list_path'], num_classes=cfg['num_classes'], train=False)
                self.val_loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])
            elif args.test:
                Dataset = LungDataset(folder=cfg['folder'], list_path=cfg['test_list_path'], num_classes=cfg['num_classes'], train=False)
                self.test_loader = DataLoader(Dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
        elif args.TumorSegment:
            if args.train or args.fast_run or args.resume:
                Dataset = TumorDataset(folder=cfg['folder'], list_path=cfg['list_path'], num_classes=cfg['num_classes'], train=True)
                self.loader = DataLoader(Dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
                Dataset = TumorDataset(folder=cfg['folder'], list_path=cfg['val_list_path'], num_classes=cfg['num_classes'], train=False)
                self.val_loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])
            elif args.test:
                Dataset = TumorDataset(folder=cfg['folder'], list_path=cfg['test_list_path'], num_classes=cfg['num_classes'], train=False)
                self.test_loader = DataLoader(Dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

        self.model = LitModel(cfg)

    def train(self):
        trainer = pl.Trainer(max_epochs=self.cfg['epochs'], check_val_every_n_epoch=5,
                             logger=self.tb_logger, log_every_n_steps=5)
        if self.args.resume:
            trainer.fit(model=self.model, train_dataloaders=self.loader, val_dataloaders=self.val_loader,
                        ckpt_path=self.args.ckpt_path)
        trainer.fit(model=self.model, train_dataloaders=self.loader, val_dataloaders=self.val_loader)

    def test(self):
        trainer = pl.Trainer(devices=1, logger=self.tb_logger)
        trainer.test(model=self.model, dataloaders=self.test_loader, ckpt_path=self.args.ckpt_path)

    def predict(self):
        trainer = pl.Trainer(devices=1)
        pred = trainer.predict(model=self.model, dataloaders=self.predict_loader, ckpt_path=self.args.ckpt_path)
        for idx, p in enumerate(pred):
            self.S.update(p, idx)
        print("Configuring all predictions to 3D segmentation...")
        pred = self.S.configure_seg()
        ct = nib.load(self.args.ct_path)
        name = self.args.ct_path.split('/')[-2]
        tools.save_niigz(ct, pred, self.cfg['default_orient'], name, self.args.saving_folder)

    def fast_run(self):
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(model=self.model, train_dataloaders=self.loader)

def main():
    args = parser()
    with open(args.yaml_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    factory = ModelFactory(args, cfg)
    if args.fast_run:
        factory.fast_run()
    elif args.train or args.resume:
        factory.train()
    elif args.test:
        factory.test()
    elif args.predict:
        factory.predict()

if __name__ == '__main__':
    main()
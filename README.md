# [VGHTC] Lung Segmentation

## Training
### Data structure (NLST)
```
-- Folder
    -- xxx
        -- xxx_CT.nii.gz
        -- xxx_Lung.nii.gz
        -- xxx_tumor.nii.gz
    -- yyy
        -- yyy_CT.nii.gz
        -- yyy_Lung.nii.gz
        -- yyy_tumor.nii.gz
    ...
```
### Preprocess
1. Convert 3D .nii.gz to 2D .npy, including CT and mask.
```=shell
$ cd data
$ python3 preprocess.py --folder_path <folder_path> --convert_3d_to_2d --json_path <label_json_path>
# e.g.
# python3 preprocess.py --folder_path /root/VGHTC/NLST_lung_Seg --convert_3d_to_2d --default_orient LPS --json_path label.json
```
2. Split dataset into training set and validation set.
```=shell
$ cd data
$ python3 preprocess.py --folder_path <folder_path> --split_data
# e.g.
# python3 preprocess.py --folder_path /root/VGHTC/No_IV_197_preprocessed/ --split_data
```

### Training
1. Train the model.
```=shell
$ python3 main.py --train --yaml_path <yaml_path>
# e.g.
# python3 main.py --train --yaml_path configs/231030.yaml
```
2. Open the Tensorboard.
```=shell
$ tensorboard --logdir lightning_logs --bind_all
```

### Testing
```=shell
$ python3 main.py --test --yaml_path <yaml_path> --ckpt_path <ckpt_path>
# e.g.
# python3 main.py --test --yaml_path configs/231030.yaml --ckpt_path checkpoints/epoch\=139-step\=91420.ckpt
```

## Inference
### Download checkpoint and example
Download the checkpoint from [Drive](http://gofile.me/6Ukc0/KCdnFlIYh) and put it in "checkpoints" folder.
And download the example (000608355C) from the same drive and put it in "example" folder.
### Data structure
```
-- CTSegment
    -- example
        -- 000608355C
            -- 000608355C_CT.nii.gz
    -- checkpoints
        -- epoch=139-step=91420.ckpt
    ...
```
### Preprocess
Convert 3D .nii.gz to 2D .npy, including CT and mask.
```=shell
$ cd data
$ python3 preprocess.py --ct_path /root/VGHTC/CTSegment/000074623G/000074623G_CT.nii.gz --convert_3d_to_2d_ct
```

### Inference
Prediction will save in "outputs" folder.
```=shell
$ python3 main.py \
    --predict \
    --ckpt_path <ckpt_path> \
    --yaml_path <yaml_path> \
    --patient_path <patient_path> \
    --ct_path <ct_path> \
    --saving_folder <saving_folder>
# e.g.
# python3 main.py \
    --predict \
    --ckpt_path checkpoints/epoch\=139-step\=91420.ckpt \
    --yaml_path configs/231030.yaml \
    --patient_path /root/VGHTC/CTSegment/000074623G/processed \
    --ct_path /root/VGHTC/CTSegment/000074623G/000074623G_CT.nii.gz \
    --saving_folder /root/VGHTC/CTSegment/outputs
```

## Evaluation
| Class| Dice  |
|:----:| :----:|
| 131  | 0.990 |
| 181  | 1.000 |
| 212  | 0.962 |
| 231  | 1.000 |
| 241  | 0.986 |

# Note
NLST dataset: 112506_2001, this data can't open by nibabel, so we can't use it.
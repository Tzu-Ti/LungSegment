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
1. Open the Tensorboard.
```=shell
$ tensorboard --logdir lightning_logs --bind_all
```
2. Train the lung segment model.
```=shell
$ python3 main.py --LungSegment --train --yaml_path <yaml_path>
# e.g.
# python3 main.py --LungSegment --train --yaml_path configs/231215.yaml
```
3. Train the tumor segment model.
```=shell
$ python3 main.py --TumorSegment --train --yaml_path <yaml_path>
# e.g.
# python3 main.py --TumorSegment --train --yaml_path configs/231217.yaml
```

## Inference
### Download checkpoint and example
Download the checkpoint from [Drive](http://gofile.me/6Ukc0/8LLuvie7y) and put it in "checkpoints" folder.
And download the example (100158_2001) from the same drive and put it in "example" folder.
### Data structure
```
-- LungSegment
    -- example
        -- 100158_2001
            -- 100158_2001_CT.nii.gz
    -- checkpoints
        -- Lung=199.ckpt
        -- Tumor=499.ckpt
    ...
```
### Preprocess
Convert 3D .nii.gz to 2D .npy, including CT and mask.
```=shell
$ cd data
$ python3 preprocess.py --ct_path /root/VGHTC/LungSegment/example/100158_2001/100158_2001_CT.nii.gz --convert_3d_to_2d_ct
```

### Testing
1. Test the lung segment model.
```=shell
$ python3 main.py --LungSegment --test --yaml_path <yaml_path> --ckpt_path <ckpt_path>
# e.g.
# python3 main.py --LungSegment --test --yaml_path configs/231215.yaml --ckpt_path checkpoints/Lung=199.ckpt
```
2. Test the tumor segment model.
```=shell
$ python3 main.py --TumorSegment --test --yaml_path <yaml_path> --ckpt_path <ckpt_path>
# e.g.
# python3 main.py --TumorSegment --test --yaml_path configs/231217.yaml --ckpt_path checkpoints/Tumor=499.ckpt
```

### Inference
Prediction will save in "outputs" folder.
```=shell
$ python3 evaluate.py \
    --predict \
    --yaml_path <yaml_path> \
    --patient_path <patient_path> \
    --ct_path <ct_path> \
    --saving_folder <saving_folder>
# e.g.
# python3 evaluate.py \
     --predict \ 
     --yaml_path configs/evaluate.yaml \
     --patient_path /root/VGHTC/LungSegment/example/100158_2001/processed \ 
     --ct_path /root/VGHTC/LungSegment/example/100158_2001/100158_2001_CT.nii.gz \
     --saving_folder Output
```

# Note
NLST dataset: 112506_2001, this data can't open by nibabel, so we can't use it.
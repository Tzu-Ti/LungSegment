__author__ = 'Titi Wei'
import argparse
import numpy as np
import glob, os
from tqdm import tqdm
import sys

import nibabel as nib

import utils

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', help='path to the folder that contains the data')
    # convert the 3D data to 2D data
    parser.add_argument('--convert_3d_to_2d', action='store_true', help='convert the 3D data to 2D data')
    parser.add_argument('--default_orient', help='default orientation of the data', default='LAS')
    parser.add_argument('--json_path', required='--convert_3d_to_2d' in sys.argv, help='path to the json file that contains the mapping label')
    # Inference
    parser.add_argument('--convert_3d_to_2d_ct', action='store_true', help='convert the 3D data to 2D data')
    parser.add_argument('--ct_path', required='--convert_3d_to_2d_ct' in sys.argv, help='path to the CT data')
    # split the data into training set and testing set
    parser.add_argument('--split_data', action='store_true', help='split the data into training set and testing set')
    # split the tumor data into training set and testing set, find the tumor data that has tumor
    parser.add_argument('--split_tumor_data', action='store_true', help='split the tumor data into training set and testing set')
    parser.add_argument('--txt_path', required='--split_tumor_data' in sys.argv, help='path to the txt file that contains the patient path')
    return parser.parse_args()

def IntensityClipping(x):
    std = x.std()
    mean = x.mean()
    MAX = mean + 3 * std
    MIN = mean - 3 * std
    x = np.clip(x, MIN, MAX)
    return x

def ComputeOrientation(init_axcodes, final_axcodes):
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_final = nib.orientations.axcodes2ornt(final_axcodes)
    ornt_transform = nib.orientations.ornt_transform(ornt_init, ornt_final)
    return ornt_transform

def reorientation(data_array, init_axcodes, final_axcodes):
    ornt_transform = ComputeOrientation(init_axcodes, final_axcodes)
    return nib.orientations.apply_orientation(data_array, ornt_transform)

def MappingLabel(json_path, seg):
    labelmap = utils.open_json(json_path)

    new = seg.copy()
    for k in tqdm(labelmap.keys()):
        if k == '0': continue
        label = labelmap[k]
        loc = np.where(seg==int(k))
        if len(loc) == 1:
            continue
        else:
            X, Y, Z = loc
        for x, y, z in zip(X, Y, Z):
            new[x, y, z] = label
    return new

def Convert3dto2d(args):
    # get the default orientation and convert it to the axcodes
    orient_axcodes = tuple(args.default_orient)

    patient_paths = glob.glob(os.path.join(args.folder_path, '*'))
    saving_root_folder = args.folder_path + '_preprocessed'

    for path in tqdm(patient_paths):
        number = os.path.basename(path)
        ct_path = os.path.join(path, '{}_CT.nii.gz'.format(number))
        lung_path = os.path.join(path, '{}_Lung.nii.gz'.format(number))
        tumor_path = os.path.join(path, '{}_tumor.nii.gz'.format(number))

        try:
            ct = nib.load(ct_path)
            lung = nib.load(lung_path)
            tumor = nib.load(tumor_path)
        except:
            print("Error:", path)
            continue

        # if the orientation of the data is not the default orientation, then reorient the data
        ct_axcodes = nib.aff2axcodes(ct.affine)
        lung_axcodes = nib.aff2axcodes(lung.affine)
        tumor_axcodes = nib.aff2axcodes(tumor.affine)
        ct = ct.get_fdata()
        lung = lung.get_fdata()
        tumor = tumor.get_fdata()
        if ct_axcodes != orient_axcodes and lung_axcodes != orient_axcodes and tumor_axcodes != orient_axcodes:
            ct = reorientation(ct, ct_axcodes, orient_axcodes)
            lung = reorientation(lung, lung_axcodes, orient_axcodes)
            tumor = reorientation(tumor, tumor_axcodes, orient_axcodes)

        # clip the intensity of the CT
        ct = IntensityClipping(ct)

        # mapping segmentation label
        json_path = args.json_path
        lung = MappingLabel(json_path, lung)
        
        # lung and tumor type convert to int
        lung = lung.astype(int)
        tumor = tumor.astype(int)

        # save the preprocessed data
        # Create the saving folder
        saving_folder = os.path.join(saving_root_folder, number)
        CT_saving_folder = os.path.join(saving_folder, 'CT')
        Lung_saving_folder = os.path.join(saving_folder, 'Lung')
        Tumor_saving_folder = os.path.join(saving_folder, 'Tumor')
        os.makedirs(CT_saving_folder, exist_ok=True)
        os.makedirs(Lung_saving_folder, exist_ok=True)
        os.makedirs(Tumor_saving_folder, exist_ok=True)

        # save the data
        for i in range(ct.shape[2]):
            CT_slice = ct[:, :, i]
            Lung_slice = lung[:, :, i]
            Tumor_slice = tumor[:, :, i]

            np.save(os.path.join(CT_saving_folder, '{:03d}.npy'.format(i)), CT_slice)
            np.save(os.path.join(Lung_saving_folder, '{:03d}.npy'.format(i)), Lung_slice)
            np.save(os.path.join(Tumor_saving_folder, '{:03d}.npy'.format(i)), Tumor_slice)

def Convert3dto2dCT(args):
    # get the default orientation and convert it to the axcodes
    orient_axcodes = tuple(args.default_orient)

    ct = nib.load(args.ct_path)

    # if the orientation of the data is not the default orientation, then reorient the data
    ct_axcodes = nib.aff2axcodes(ct.affine)
    ct = ct.get_fdata()
    if ct_axcodes != orient_axcodes:
        ct = reorientation(ct, ct_axcodes, orient_axcodes)

    # clip the intensity of the CT
    ct = IntensityClipping(ct)

    # save the preprocessed data
    # Create the saving folder
    saving_folder = os.path.join(os.path.dirname(args.ct_path), "processed")
    CT_saving_folder = os.path.join(saving_folder, 'CT')
    os.makedirs(CT_saving_folder, exist_ok=True)

    # save the data
    for i in range(ct.shape[2]):
        CT_slice = ct[:, :, i]

        np.save(os.path.join(CT_saving_folder, '{:03d}.npy'.format(i)), CT_slice)

def SplitData(args):
    patient_paths = glob.glob(os.path.join(args.folder_path, '*'))
    
    length = len(patient_paths)
    trainList = patient_paths[:int(length*0.8)]
    testList = patient_paths[int(length*0.8):]

    with open('trainList.txt', 'w') as f:
        for path in trainList:
            f.write(os.path.basename(path) + '\n')
    with open('testList.txt', 'w') as f:
        for path in testList:
            f.write(os.path.basename(path) + '\n')

    print("Training list and testing list are written to trainList.txt and testList.txt")

from utils import open_txt
def SplitTumorData(txt_path):
    patients = open_txt(txt_path)

    tumor_candidate_paths = []
    for patient in tqdm(patients):
        patient_path = os.path.join(args.folder_path, patient)
        tumor_paths = glob.glob(os.path.join(patient_path, 'Tumor', '*.npy'))
        for path in tumor_paths:
            tumor_seg = np.load(path)
            if tumor_seg.sum() > 0:
                print(path)
                tumor_candidate_paths.append(path)

    with open(f'tumor_{txt_path}', 'w') as f:
        for path in tumor_candidate_paths:
            f.write(path + '\n')

if __name__ == '__main__':
    args = parse()
    if args.convert_3d_to_2d:
        Convert3dto2d(args)
    elif args.split_data:
        SplitData(args)
    elif args.convert_3d_to_2d_ct:
        Convert3dto2dCT(args)
    elif args.split_tumor_data:
        SplitTumorData(args.txt_path)
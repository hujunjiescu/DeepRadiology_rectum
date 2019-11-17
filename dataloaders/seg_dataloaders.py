import os, pdb
import pydicom
import SimpleITK as sitk
import numpy as np
import operator
from PIL import Image, ImageOps
import warnings
import cv2
import collections
from torch.utils.data import Dataset
from tqdm import tqdm

def slice_order(sample_path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    slices = []
    for filename in os.listdir(sample_path):
        if filename.startswith("CT"):
            f = pydicom.read_file(os.path.join(sample_path, filename))
            slices.append(f)
            
    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return [slice_info[0] for slice_info in ordered_slices]

class Seg_dataset_online(Dataset):
    def __init__(self, data_root, 
                        sample_txt,
                        mask_dict,
                        transforms,
                        txt_include_mask = True,
                        CT_prefixs = ["CT.", "CT"],
                        foreground_only = False,
                        img_w = 512, 
                        img_h = 512
                        ):
        Record = collections.namedtuple('Record', ['sample_name', 'sample_path',  'image_raw_name', 'image_name', 'source_masks'])
        self.mask_dict = mask_dict
        self.transforms = transforms
        self.img_w, self.img_h = img_w, img_h
        self.records = []
        if txt_include_mask:
            sample_masks = self._resolve_samplemask_txt(sample_txt)
        else:
            source_masks = []
            for mask_names in mask_dict.values():
                source_masks += mask_names
            sample_masks = self._resolve_sample_txt(sample_txt, source_masks)
        
        for (sample_name, source_masks) in tqdm(sample_masks.items()):
            sample_path = os.path.join(data_root, sample_name)
            ct_raw_names = slice_order(sample_path)
            for raw_name in ct_raw_names:
                for CT_prefix in CT_prefixs:
                    img_name = "{}{}.dcm".format(CT_prefix, raw_name)
                    if os.path.exists(os.path.join(sample_path, img_name)): break
                record = Record(sample_name, sample_path, raw_name, img_name, source_masks)
                _, combined_mask = self._resolve_record(record, source_masks)
                if foreground_only and np.sum(combined_mask) == 0: 
                    continue
                else:
                    self.records.append(record)
    
    def _resolve_sample_txt(self, sample_txt, mask_names):
        fp = open(sample_txt, "r")
        lines = fp.readlines()
        fp.close()
        sample_masks = {}
        for line in lines:
            line = line.strip()
            if line != "":
                sample_name = line.split("\t")[0]
                sample_masks[sample_name] = mask_names
        return sample_masks
    
    def _resolve_samplemask_txt(self, sample_txt):
        fp = open(sample_txt, "r")
        lines = fp.readlines()
        fp.close()
        sample_masks = {}
        for line in lines:
            line = line.strip()
            if line != "":
                sample_name = line.split("\t")[0]
                mask_names = line.split("\t")[1:]
                sample_masks[sample_name] = mask_names
        return sample_masks
    
    def _resolve_record(self, record, source_masks):
        img_path = os.path.join(record.sample_path, record.image_name)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[0]
        mask_list, labels = [], []
        for (mask_label, target_masks) in self.mask_dict.items():
            for source_mask in source_masks:
                if source_mask in target_masks:
                    mask = self._get_mask(record.sample_path, record.image_raw_name, source_mask, mask_label, self.img_w, self.img_h)
                    if mask is not None:
                        mask_list.append(mask)
                        labels.append(mask_label)
        combined_mask = self._combine_masks(mask_list, labels)
        return img, combined_mask
    
    def _get_mask(self, sample_path, ct_raw_name, mask_name,  mask_label, img_w, img_h):
        
        mask_path = os.path.join(sample_path, "mask", mask_name)
        if not os.path.exists(mask_path): 
            return None
        
        target_mask_path = os.path.join(mask_path, ct_raw_name + ".bmp")
        if os.path.exists(target_mask_path):
            mask = cv2.imread(target_mask_path)[:, :, 0]
            np.place(mask, mask == 255, mask_label)
            return mask
        else:
            return None
    
    def _combine_masks(self, mask_list, labels):
        """combine all masks into one in an sample"""
        combined_mask = np.zeros((self.img_w, self.img_h), dtype=np.uint8)
        if len(mask_list) == 0:
            return combined_mask
        
        for i in range(len(mask_list)):
            np.place(combined_mask, mask_list[i] == labels[i], labels[i])
        return combined_mask
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        img, mask = self._resolve_record(self.records[idx], self.records[idx].source_masks)
        sample = {'image': img, 'label': mask, 'sample_name': self.records[idx].sample_name, 'image_name':self.records[idx].image_raw_name}
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
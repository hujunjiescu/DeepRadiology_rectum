
# coding: utf-8

# In[1]:


import os, sys, pdb, shutil, random, math, datetime, operator
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm

from dataloaders.seg_dataloaders import Seg_dataset_online
from networks.segmentation.deeplab_resnet import DeepLabv3_plus_resnet
from networks.segmentation.res_unet import ResUNet50
from networks.segmentation.DDCNN import DDCNN

from networks_cfgs.deeplab import deeplab_resnet50
from networks_cfgs.res_unet import res_unet50_cfg
from config import Config

from dataloaders import seg_transforms as tr
from losses import CrossEntropy2D
import tasks.rectum_tasks as rectum_tasks

def get_logger(log_path):
    parent_path = os.path.dirname(log_path)  # get parent path
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger

def log_best_metric(metric_list, cur_epoch_idx, logger, state, save_path, save_model=True, metric = "Dice"):
    if len(metric_list) == 0:
        return
    else:
        best_idx = np.argmax(metric_list)
        best_metric = metric_list[best_idx]
        if best_idx == cur_epoch_idx:
            logger.info("Epoch: %d, Validation %s improved to %.4f"%(cur_epoch_idx, metric, best_metric))
            if save_model:
                dir_path = os.path.dirname(save_path)  # get parent path
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(state, save_path)
                logger.info("Model saved in file: %s"%(save_path))
        else:
            logger.info("Epoch: %d, Validation %s didn't improve. Best is %.4f in epoch %d"%(cur_epoch_idx, metric, best_metric, best_idx))

def train(model, device, data_loader, criterion, optimizer, epoch, writer, metric_idxs):
    model.train()
    losses = []
    volume_dict = {}
    for metric_idx in metric_idxs:
        volume_dict[metric_idx] = np.array([0, 0, 0, 0])
        
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, sample_batched in enumerate(data_loader):
            inputs, labels = sample_batched['image'], sample_batched['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device).long()
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()
            optimizer.zero_grad()
            
            predictions = torch.max(outputs, 1)[1].cpu().numpy()
            labels = labels.cpu().numpy()
            
            for metric_idx in metric_idxs:
                volume_dict[metric_idx] += seg_volume(predictions, labels, metric_idx)
            
            pbar.update(1)
            pbar.set_description("Epoch %d, Batch %d/%d, Train loss: %.4f"%(epoch, batch_idx+1, len(data_loader), np.mean(losses)))
    
    ave_loss = np.mean(losses)
    writer.add_scalar('train/epoch_loss', ave_loss, epoch)
    dices, TPVFs, PPVs = segmentation_metrics(volume_dict, metric_idxs)
    return ave_loss, dices, TPVFs, PPVs


def validate(model, device, data_loader, criterion, epoch, writer, metric_idxs):
    losses = []
    model.eval()
    volume_dict = {}
    for metric_idx in metric_idxs:
        volume_dict[metric_idx] = np.array([0, 0, 0, 0])
    
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(data_loader)):
            inputs, labels = sample_batched['image'], sample_batched['label']
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            
            outputs,_ = model(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            predictions = torch.max(outputs, 1)[1].cpu().numpy()
            labels = labels.cpu().numpy()
            
            for metric_idx in metric_idxs:
                volume_dict[metric_idx] += seg_volume(predictions, labels, metric_idx)
                
    ave_loss = np.mean(losses)
    writer.add_scalar('test/epoch_loss', ave_loss, epoch)
    dices, TPVFs, PPVs = segmentation_metrics(volume_dict, metric_idxs)
                
    return ave_loss, dices, TPVFs, PPVs


def seg_volume(batch_output, batch_label, metric_idx):
    
    mask_o = (batch_output == metric_idx)
    mask_y = (batch_label == metric_idx)
    union = (mask_o * mask_y).sum()
    inter = mask_o.sum() + mask_y.sum()
    v_y = mask_y.sum()
    v_o = mask_o.sum()

    return np.array([union, inter, v_y, v_o])

def segmentation_metrics(volume_dict, metric_idxs = [1, 2]):
    dices, TPVFs, PPVs = [], [], []
    for metric_idx in metric_idxs:
        [union, inter, v_y, v_o] = list(volume_dict[metric_idx])
            
        dice = 0 if inter == 0 else round(float(2*union) / inter, 5)
        TPVF = 0 if v_y == 0 else round(float(union) / v_y, 5)
        PPV = 0 if v_o == 0 else round(float(union) / v_o, 5)
        dices.append(dice)
        TPVFs.append(TPVF)
        PPVs.append(PPV)
        
    return dices, TPVFs, PPVs            
            

if __name__ == '__main__':    
    config = Config()

    log_path = os.path.join('logs', config.network, '{}.log'.format(config.suffix))
    checkpoint_path = os.path.join('checkpoint', config.network, config.suffix)
    summary_path = os.path.join("summaries", config.network, config.suffix)

    logger = get_logger(log_path)
    writer = SummaryWriter(summary_path)
    logger.info(config.__dict__)

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    logger.info("Random Seed: {}".format(config.manualSeed))
    np.random.seed(config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)

    train_tr = transforms.Compose([
                tr.Filter_HU(config.HU_min, config.HU_max),
                tr.Normalize_maxmin(config.HU_min, config.HU_max),
                tr.Arr2image(),
                tr.CenterCrop([config.h, config.w]),
                tr.RandomHorizontalFlip(),
                tr.RandomSized([config.h, config.w], config.scale_min, config.scale_max), # h, w
                tr.RandomRotate(config.rotation),
                tr.ToTensor()
            ])

    test_tr = transforms.Compose([
                tr.Filter_HU(config.HU_min, config.HU_max),
                tr.Normalize_maxmin(config.HU_min, config.HU_max),
                tr.Arr2image(),
                tr.CenterCrop([config.h, config.w]),
                tr.ToTensor()
            ])

    train_dataset = Seg_dataset_online(config.data_root,
                                config.train_txt,
                                config.label_dict, 
                                foreground_only = True,
                                txt_include_mask = False,
                                transforms = train_tr)

    test_dataset = Seg_dataset_online(config.data_root,
                                config.test_txt,
                                config.label_dict, 
                                foreground_only = True,
                                txt_include_mask = False,
                                transforms = test_tr)

    trainset_loader = DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=config.num_workers)
    testset_loader = DataLoader(test_dataset, batch_size=config.test_batch, shuffle=False, num_workers=config.num_workers)
    logger.info("Number of training images: {}, test images: {}".format(len(train_dataset), len(test_dataset)))
    # In[6]:

    model = globals()[config.network](config.in_channels, config.num_classes, config.net_config)
    
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.wd)
    lr_lambda = lambda epoch: (1 - float(epoch) / config.nepoch)** 0.9
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    gpus = range(len(config.gpus.split(",")))
    if len(gpus) > 1:
        model = nn.DataParallel(model, gpus)
    device = torch.device("cuda:{}".format(gpus[0]))
    model.to(device)

    if config.criterion == "cross_entropy":
        criterion = CrossEntropy2D(weight=config.class_weight, size_average=True, batch_average=True)
    else:
        raise("Unknown criterion: {}".format(config.criterion))

    metric_list = []
    for epoch in range(config.nepoch):
        lr_scheduler.step()
        logger.info("Epoch: %d, Learning rate: %.10f"%(epoch, lr_scheduler.get_lr()[0]))
        train_loss, train_dices, train_TPVFs, train_PPVs = train(model, device, trainset_loader, criterion, optimizer, epoch, writer, config.metric_indexs)
        logger.info("Epoch: %d, Train Loss: %.4f, \nDices: %s \nTPVF: %s \nPPV: %s"
                    %(epoch, train_loss, train_dices, train_TPVFs, train_PPVs))

        test_loss, test_dices, test_TPVFs, test_PPVs = validate(model, device, testset_loader, criterion, epoch, writer, config.metric_indexs)
        avg_score = round(np.mean(test_dices), 5)
        metric_list.append(avg_score)
        logger.info("Epoch: %d, Test Loss: %.4f, \nDices: %.4f, %s \nTPVF: %s \nPPV: %s"
                    %(epoch, test_loss, avg_score, test_dices, test_TPVFs, test_PPVs))

        log_best_metric(metric_list, epoch, logger, 
                        {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                         '{}/epoch{}.pth'.format(checkpoint_path, epoch),
                        save_model=True,
                        metric = "Dice score")
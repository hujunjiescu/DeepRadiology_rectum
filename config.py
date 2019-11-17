class Config(object):
    def __init__(self):
        
        self.train_batch = 20 # 32
        self.test_batch = 10
        self.data_root = "./data/rectum_all/"
        self.train_txt = "./datalist/rectum/temp.txt"
        self.test_txt = "./datalist/rectum/temp.txt"
        
        self.nepoch = 300
        self.HU_max = 390 
        self.HU_min = -310 
        
        self.h = 512
        self.w = 512
        self.lr = 0.01
        self.in_channels = 1
        self.num_classes = 2
        self.class_weight = [1, 2]
        self.wd = 5e-4
        self.momentum = 0.9
         
        self.network = 'ResUNet50'
        self.net_config = res_unet50_cfg #res_unet50_regularize_cfg([0.5, 0.5, 0.5])
        
        self.criterion = "cross_entropy" # cross_entropy | dice
        self.scale_min = 0.5
        self.scale_max = 2.0
        self.rotation = 15
        
        self.suffix = "rectum_CTV_lr0.01_batch20_weight4" #rectum_OAR_lr0.01_Femoral_All
        self.checkpoint = None
        self.label_dict = rectum_tasks.CTV
        self.metric_indexs = [1]
        
        self.gpus = "0, 1"
        self.num_workers = 8
        
        self.manualSeed = 1
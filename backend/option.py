# option.py
import os
abspath = os.path.abspath(os.getcwd())

class Options:
    def __init__(self):
        self.modality = 'MIX2'
        self.rgb_list = os.path.join(abspath, "MultiModel/list/rgb.list")
        self.flow_list = os.path.join(abspath, "MultiModel/list/flow.list")
        self.audio_list = os.path.join(abspath, "MultiModel/list/audio.list")
        self.test_rgb_list = os.path.join(abspath, "MultiModel/list/my_RGBtest.list")
        self.test_flow_list = os.path.join(abspath, "MultiModel/list/my_Flowtest.list")
        self.test_audio_list = os.path.join(abspath, "MultiModel/list/my_Audiotest.list")
        self.gt = 'gtMulti.npy'
        self.gpus = 0
        self.lr = 0.0001
        self.batch_size = 128
        self.workers = 4
        self.model_name = 'wsanodetV5_'
        self.pretrained_ckpt = None
        self.feature_size = 1024 + 128
        self.num_classes = 7
        self.dataset_name = 'XD-Violence'
        self.max_seqlen = 200
        self.max_epoch = 50
        self.weights = 'Normal'
        self.online_mode = 'Binary'
        self.optimizer = 'Adam'

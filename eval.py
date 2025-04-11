from evaluation.evaler import Evaler
from lib.config import cfg, cfg_from_file

import os
import sys
import argparse
import torch
import models
import lib.utils as utils
import warnings
warnings.filterwarnings("ignore")

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        # Check available devices
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.setup_network()
        self.val_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        )
        self.test_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(model).cuda()
        else:
            self.model = model.to(self.device)
        if self.args.resume > 0:
            # Load model weights
            model_path = self.snapshot_path("student_model", self.args.resume)
            print(f"Loading model: {model_path}")
            
            success = utils.load_model_weights(self.model, model_path, self.device)
            if success:
                print("Model loaded successfully")
            else:
                print("Failed to load model weights")
    
    def verify_model_weights(self):
        """Verify if weights are loaded correctly"""
        # Get the actual model instance
        if hasattr(self.model, 'module'):
            target_model = self.model.module
        else:
            target_model = self.model
            
        # Check if weights contain NaN or inf
        has_errors = False
        for name, param in target_model.named_parameters():
            if torch.isnan(param).any():
                print(f"Parameter {name} contains NaN values")
                has_errors = True
            if torch.isinf(param).any():
                print(f"Parameter {name} contains infinite values")
                has_errors = True
        
        if not has_errors:
            print("Weight verification passed, all parameters are normal")
        else:
            print("Warning: Weight verification failed, some parameters are abnormal")

    def eval(self, epoch):
        res = self.val_evaler(self.model, 'val_' + str(epoch))
        print('######## Epoch(VAL) ' + str(epoch) + ' ########')
        print(str(res))
        res = self.test_evaler(self.model, 'test_' + str(epoch))
        print('######## Epoch(TEST) ' + str(epoch) + ' ########')
        print(str(res))

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(args.resume)

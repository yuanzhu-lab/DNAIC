from lib.config import cfg, cfg_from_file

import os
import sys
import numpy as np
import torch
import tqdm
import json
import argparse
import lib.utils as utils
import datasets.data_loader as data_loader
import models
import warnings
warnings.filterwarnings("ignore")

class OnlineTester(object):
    def __init__(
        self,
        eval_ids,
        feats_folder
    ):
        super(OnlineTester, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        with open(eval_ids, 'r') as f:
            self.ids2path = json.load(f)
            self.eval_ids = np.array(list(self.ids2path.keys()))
        
        self.eval_loader = data_loader.load_val(eval_ids, feats_folder, test=1)

    def __call__(self, model, rname, start, end):
        model.eval()
        
        all_time = .0
        
        # Get the actual model instance
        if hasattr(model, 'module'):
            target_model = model.module
        else:
            target_model = model

        results = []
        with torch.no_grad():
            for _, (indices, att_feats) in enumerate(tqdm.tqdm(self.eval_loader)):
                ids = self.eval_ids[indices]
                if torch.cuda.is_available():
                    att_feats = att_feats.cuda()
                elif torch.backends.mps.is_available():
                    att_feats = att_feats.to("mps")
                
                att_feats = target_model.inference_pre(att_feats)
                
                if torch.cuda.is_available():
                    start.record()
                    out = target_model.inference(att_feats)
                    end.record()
                    torch.cuda.synchronize()
                    all_time += start.elapsed_time(end)
                else:
                    import time
                    start_time = time.time()
                    out = target_model.inference(att_feats)
                    all_time += (time.time() - start_time) * 1000
                
                prob, seq = torch.max(out, -1)
                
                sents = utils.decode_sequence(self.vocab, seq.data)
                sents = utils.clip_chongfu(sents)
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)
            print(f"Average inference time: {all_time/len(self.eval_loader)} ms")    
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def snapshot_path(name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    model = models.create(cfg.MODEL.TYPE)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        start = None
        end = None
    else:
        device = torch.device("cpu")
        model = model.to(device)
        start = None
        end = None
    
    print(f"Using device: {device}")
    
    if args.resume > 0:
        checkpoint_path = snapshot_path("student_model", args.resume)
        print(f"Loading model: {checkpoint_path}")
        success = utils.load_model_weights(model, checkpoint_path, device)
        if not success:
            print("Failed to load model weights. Exiting.")
            sys.exit(1)
    
    print(f"Loading checkpoint {args.resume}")
    train_ids = './mscoco/misc/ids2path_json/coco_train_ids2path.json'
    eval_ids = './mscoco/misc/ids2path_json/coco_val_ids2path.json'
    test_ids = './mscoco/misc/ids2path_json/coco_test_ids2path.json'
    test4w_ids = './mscoco/misc/ids2path_json/coco_test4w_ids2path.json'
    att_feats = './mscoco/feature/coco2014'
    tester = OnlineTester(test4w_ids, att_feats)
    tester(model, "test4w", start, end)
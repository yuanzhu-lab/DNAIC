import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
from losses.distillation_loss import DistillationLoss
import warnings
warnings.filterwarnings("ignore")

from transformers import BertModel
BertModel.from_pretrained("bert-base-uncased")

class DistillationTrainer(object):
    def __init__(self, args):
        super(DistillationTrainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(cfg.SEED)

        # Check available devices
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # MPS does not support distributed training
        self.distributed = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.distributed = True
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

        self.rl_stage = False
        self.setup_logging()
        self.setup_dataset()
        self.setup_networks()
        self.setup_distillation()
        self.val_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE,
            image_size = 384
        )
        self.test_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE,
            image_size = 384
        )
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_dataset(self):
        self.student_coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
            image_size = 384
        )
        
        self.teacher_coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
            image_size = 256
        )
        
        self.coco_set = self.student_coco_set

    def setup_loader(self, epoch):
        self.student_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.student_coco_set)
        
        self.teacher_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.teacher_coco_set)
        
        self.training_loader = self.student_loader

    def setup_networks(self):
        # Set up teacher model (autoregressive)
        self.logger.info("Starting to create teacher model")
        teacher_model = models.create(cfg.MODEL.TEACHER_TYPE)
        self.logger.info(f"Teacher model type: {cfg.MODEL.TEACHER_TYPE}")
        
        if self.distributed:
            self.logger.info("Using Distributed Data Parallel (DDP) to wrap teacher model")
            teacher_model = teacher_model.to_empty(device=self.device)
            self.teacher_model = torch.nn.parallel.DistributedDataParallel(
                teacher_model,
                device_ids = [self.args.local_rank],
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.logger.info("Using Data Parallel (DP) to wrap teacher model")
            teacher_model = teacher_model.to_empty(device=self.device)
            self.teacher_model = torch.nn.DataParallel(teacher_model)
        
        # Load teacher model weights - handled in ARTeacher class
        if os.path.exists(cfg.MODEL.TEACHER_WEIGHTS):
            self.logger.info(f"Teacher model weight file exists at: {cfg.MODEL.TEACHER_WEIGHTS}")
            print("Teacher model weight file exists at:", cfg.MODEL.TEACHER_WEIGHTS)
        else:
            self.logger.warning(f"Warning: Teacher model weight file {cfg.MODEL.TEACHER_WEIGHTS} does not exist!")
            print(f"Warning: Teacher model weight file {cfg.MODEL.TEACHER_WEIGHTS} does not exist!")
        
        self.teacher_model.eval()  # Set to evaluation mode
        self.logger.info("Teacher model set to evaluation mode")

        # Set up student model (non-autoregressive)
        self.logger.info("Starting to create student model")
        student_model = models.create(cfg.MODEL.STUDENT_TYPE)
        self.logger.info(f"Student model type: {cfg.MODEL.STUDENT_TYPE}")
        
        if self.distributed:
            self.logger.info("Using Distributed Data Parallel (DDP) to wrap student model")
            self.student_model = torch.nn.parallel.DistributedDataParallel(
                student_model.to(self.device),
                device_ids = [self.args.local_rank],
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.logger.info("Using Data Parallel (DP) to wrap student model")
            self.student_model = torch.nn.DataParallel(student_model).to(self.device)

        self.load_epoch = -1
        self.load_iteration = -1
        if self.args.resume > 0:
            checkpoint_path = self.snapshot_path("student_model", self.args.resume)
            self.logger.info(f"Loading student model checkpoint: {checkpoint_path}")
            print(f"Loading student model checkpoint: {checkpoint_path}")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.logger.info("Successfully loaded student model checkpoint file")
                
                # Handle checkpoint format
                if "model" in checkpoint:
                    self.logger.info("Extracting model part from complete checkpoint")
                    self.student_model.load_state_dict(checkpoint["model"])
                else:
                    self.logger.info("Loading model weights directly")
                    self.student_model.load_state_dict(checkpoint)
                
                self.logger.info("Student model loaded successfully")
                print("Student model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading student model: {str(e)}")
                print(f"Error loading student model: {str(e)}")

            if self.args.load_epoch:
                self.load_epoch = self.args.resume - 1
                self.load_iteration = int(self.args.resume * 113287 / cfg.TRAIN.BATCH_SIZE)
                self.logger.info(f"Set loading starting point: epoch={self.load_epoch}, iteration={self.load_iteration}")

        self.logger.info("Initializing optimizer")
        self.optim = Optimizer(self.student_model, self.load_iteration + 1)
        self.logger.info(f"Optimizer type: {cfg.SOLVER.TYPE}, Learning rate: {cfg.SOLVER.BASE_LR}, Starting iteration: {self.load_iteration + 1}")

    def setup_distillation(self):
        # Get vocabulary sizes for teacher and student models
        teacher_vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        student_vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        
        # If different vocabulary sizes are specified in config, use those
        if hasattr(cfg.MODEL, 'TEACHER_VOCAB_SIZE'):
            teacher_vocab_size = cfg.MODEL.TEACHER_VOCAB_SIZE + 1
            
        if hasattr(cfg.MODEL, 'STUDENT_VOCAB_SIZE'):
            student_vocab_size = cfg.MODEL.STUDENT_VOCAB_SIZE + 1
            
        print(f"Teacher model vocabulary size: {teacher_vocab_size}, Student model vocabulary size: {student_vocab_size}")
        
        self.distillation_criterion = DistillationLoss(
            temperature=cfg.DISTILLATION.TEMPERATURE,
            alpha=cfg.DISTILLATION.ALPHA,
            teacher_vocab_size=teacher_vocab_size,
            student_vocab_size=student_vocab_size
        ).to(self.device)

    def forward_distillation(self, student_att_feats, target_seq, input_seq=None, teacher_att_feats=None):
        with torch.no_grad():
            teacher_logits = self.teacher_model(teacher_att_feats, input_seq)
        
        student_logits = self.student_model(student_att_feats)
        
        loss = self.distillation_criterion(student_logits, teacher_logits, target_seq)
        return loss

    def train(self):
        self.logger.info("Starting training process")
        self.student_model.train()
        self.teacher_model.eval()
        self.logger.info("Student model set to training mode, teacher model set to evaluation mode")

        iteration = self.load_iteration + 1
        self.logger.info(f"Starting training from iteration {iteration}")
        
        for epoch in range(self.load_epoch + 1, cfg.SOLVER.MAX_EPOCH):
            self.logger.info(f"====== Starting epoch {epoch} ======")
            if epoch >= cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
                self.logger.info("Entering reinforcement learning stage")
                
            self.setup_loader(epoch)
            self.logger.info(f"Data loader ready, total {len(self.training_loader)} batches")
            
            teacher_data_iter = iter(self.teacher_loader)
            
            running_loss = .0
            with tqdm.tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.student_loader)) as pbar:
                for batch_idx, (indices, input_seq, target_seq, student_att_feats) in enumerate(self.student_loader):
                    try:
                        _, _, _, teacher_att_feats = next(teacher_data_iter)
                    except StopIteration:
                        teacher_data_iter = iter(self.teacher_loader)
                        _, _, _, teacher_att_feats = next(teacher_data_iter)
                    
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    student_att_feats = student_att_feats.to(self.device)
                    teacher_att_feats = teacher_att_feats.to(self.device)

                    loss = self.forward_distillation(student_att_feats, target_seq, input_seq, teacher_att_feats)
                    self.optim.zero_grad()
                    loss.backward()
                    utils.clip_gradient(self.optim.optimizer, self.student_model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    self.optim.step()
                    self.optim.scheduler_step('Iter')

                    running_loss += loss.item()
                    avg_loss = running_loss / (batch_idx + 1)
                    pbar.set_postfix(loss='%.2f' % avg_loss)
                    pbar.update()
                    
                    # Log every 100 batches
                    if (batch_idx + 1) % 100 == 0:
                        self.logger.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(self.student_loader)}, Loss: {avg_loss:.4f}")
                    
                    iteration += 1
                    if self.distributed:
                        dist.barrier()
            
            # Record average loss for each epoch
            self.logger.info(f"Epoch {epoch} completed, average loss: {running_loss / len(self.student_loader):.4f}")
            
            self.save_model(epoch)
            val = self.eval(epoch)
            self.logger.info(f"Epoch {epoch} validation completed, validation score: {val}")
            self.optim.scheduler_step('Epoch', val)
            
            if self.distributed:
                dist.barrier()
                
        self.logger.info("Training completed!")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.student_model.state_dict(), self.snapshot_path("student_model", epoch+1))

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        val_res = self.val_evaler(self.student_model, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        test_res = self.test_evaler(self.student_model,'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Image Captioning with Knowledge Distillation')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument("--load_epoch", action='store_true')

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

    trainer = DistillationTrainer(args)
    trainer.train() 
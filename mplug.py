from lib.config import cfg
from mPLUG.models.model_caption_mplug import MPLUG
from mPLUG.models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from mPLUG.models.visual_transformers import initialize_clip
from mPLUG.models.predictor import TextGenerator
from transformers import BertTokenizer
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp
from PIL import Image

import os
import sys
import logging
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
mplug_path = os.path.join(current_dir, 'mPLUG')
if mplug_path not in sys.path:
    sys.path.append(mplug_path)

class TeacherMPLUG(MPLUG):
    def __init__(self):
        """Initialize TeacherMPLUG model"""
        nn.Module.__init__(self)
        
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.info("Initializing TeacherMPLUG model")
        
        # Initialize gradient_checkpointing attribute
        self.gradient_checkpointing = False
        
        # Load tokenizer
        tokenizer_path = cfg.MODEL.MPLUG.TOKENIZER_PATH if hasattr(cfg.MODEL, 'MPLUG') and hasattr(cfg.MODEL.MPLUG, 'TOKENIZER_PATH') else "bert-base-uncased"
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"Using tokenizer: {tokenizer_path}")
        except Exception as e:
            local_tokenizer_path = os.path.join(current_dir, 'pretrained_models', 'bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)
            self.logger.info(f"Loaded tokenizer from local path: {local_tokenizer_path}")
        
        # Set vocabulary size
        self.vocab_size = (cfg.MODEL.TEACHER_VOCAB_SIZE if hasattr(cfg.MODEL, 'TEACHER_VOCAB_SIZE') else cfg.MODEL.VOCAB_SIZE) + 1
        
        # Build configuration and initialize model components
        self.config = self.build_config()
        self.module_setting(self.config)
        
        # Initialize visual encoder
        self.visual_encoder, _ = initialize_clip(self.config)
        
        # Initialize text encoder, fusion encoder and text decoder
        self.text_encoder = BertModel.from_pretrained(self.config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionModel.from_pretrained(self.config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)
        self.text_decoder = BertPrefixModel.from_pretrained(self.config['text_decoder'], config=self.config_decoder)
        
        # Initialize beam search generator
        self.beam_generator = TextGenerator(self.config, self.text_decoder)
        
        # Load pre-trained weights
        if hasattr(cfg.MODEL, 'TEACHER_WEIGHTS') and cfg.MODEL.TEACHER_WEIGHTS:
            self.load_pretrained_weights()
    
    def module_setting(self, config):
        """Module settings"""
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        
        self.use_checkpoint = config.get("use_checkpoint", True)
        self.logger.info(f"use_checkpoint: {self.use_checkpoint}")
    
    def build_config(self):
        """Build mPLUG configuration"""
        # Default configuration
        config = {
            'text_encoder': 'bert-base-uncased',
            'text_decoder': 'bert-base-uncased',
            'bert_config': os.path.join(mplug_path, 'configs/config_bert.json'),
            'vision_width': 768,
            'use_checkpoint': True,
            'image_res': 256,
            'patch_size': 16,
            'max_words': 30,
            'max_tokens': 30,
            'beam_size': 5,
            'min_length': 1,
            'max_length': 20,
            'vit': 'base',
            'clip_name': 'ViT-B-16',
            'distill': True,
            'clip_model_path': cfg.MODEL.MPLUG.CLIP_MODEL_PATH if hasattr(cfg.MODEL, 'MPLUG') and hasattr(cfg.MODEL.MPLUG, 'CLIP_MODEL_PATH') else "pretrained_models/vit/ViT-B-16.tar"
        }
        
        # Override default configuration with cfg configuration
        if hasattr(cfg.MODEL, 'MPLUG'):
            for key in config:
                if hasattr(cfg.MODEL.MPLUG, key.upper()):
                    config[key] = getattr(cfg.MODEL.MPLUG, key.upper())
        
        self.logger.info(f"mPLUG configuration: {config}")
        return config
    
    def forward(self, att_feats, seq=None, train=True, out_size=5):
        """Forward propagation function"""
        # Convert image features to correct data type
        image = att_feats.to(dtype=next(self.parameters()).dtype)
        
        # Get image features and attention mask
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.gradient_checkpointing)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        if train and seq is not None:
            # Training mode - use teacher forcing
            answer = seq
            
            # Create attention mask
            attention_mask = (answer != self.tokenizer.pad_token_id).long()
            
            # Create target tensor, set padding positions to -100
            answer_targets = answer.clone()
            answer_targets[answer == self.tokenizer.pad_token_id] = -100
            
            # Call text decoder
            answer_output = self.text_decoder(
                answer, 
                attention_mask=attention_mask, 
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,                  
                labels=answer_targets,
                return_dict=True,   
                reduction='none'
            )
            
            # Return logits for distillation
            return answer_output.logits
        else:
            # Inference mode - generate text
            topk_ids, topk_probs = self.generation(image_embeds, image_atts, out_size=out_size)
            return topk_ids, topk_probs
    
    def beam_search(self, image, answer=None, train=True, out_size=5):
        """Perform beam search generation"""
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.gradient_checkpointing)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        topk_ids, topk_probs = self.generation(image_embeds, image_atts, out_size=out_size)
        return topk_ids, topk_probs
    
    def generation(self, question_states, question_atts, out_size=1):
        """Generate text"""
        encoder_inputs = [question_states, question_atts]
        topk_ids, topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs, out_size=out_size)  
        return topk_ids, topk_probs
    
    def load_pretrained_weights(self):
        """Load pre-trained weights"""
        self.logger.info(f"Loading pre-trained weights: {cfg.MODEL.TEACHER_WEIGHTS}")
        try:
            checkpoint = torch.load(cfg.MODEL.TEACHER_WEIGHTS, map_location='cpu', weights_only=True)
            
            # Get state dict
            state_dict = checkpoint.get('model', checkpoint.get('module', checkpoint))
            
            model_state_dict = {}
            missing_keys = []
            
            # Extract visual encoder and text decoder weights
            for key, value in state_dict.items():
                if key.startswith(('visual_encoder', 'text_decoder')):
                    if key in self.state_dict():
                        if self.state_dict()[key].shape == value.shape:
                            model_state_dict[key] = value
                        else:
                            missing_keys.append(f"{key} (shape mismatch)")
            
            # Load weights
            self.load_state_dict(model_state_dict, strict=False)
            
            self.logger.info(f"Successfully loaded {len(model_state_dict)} parameters")
            self.logger.info("Pre-trained weights loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading weights: {str(e)}")
    
    # Distillation related methods
    def init_distill(self, config):
        """Initialize distillation model"""
        self.distill = config['distill']
        if self.distill:
            self.logger.info("Initializing distillation model")
            self.visual_encoder_m, _ = initialize_clip(config)
            self.text_encoder_m = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
            self.fusion_encoder_m = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)
            self.text_decoder_m = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)
            
            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.fusion_encoder, self.fusion_encoder_m],
                [self.text_decoder, self.text_decoder_m],
            ]
            
            if self.config_encoder.hidden_size != config['vision_width']:
                self.visn_fc_m = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend([
                    [self.visn_fc, self.visn_fc_m], 
                    [self.visn_layer_norm, self.visn_layer_norm_m]
                ])
                
            self.copy_params()
            self.momentum = 0.995
            self.logger.info("Distillation model initialization completed")

    @torch.no_grad()
    def copy_params(self):
        """Copy parameters to momentum model"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False
                
    @torch.no_grad()
    def _momentum_update(self):
        """Use momentum to update model"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum) 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((cfg.MODEL.MPLUG.IMAGE_RES, cfg.MODEL.MPLUG.IMAGE_RES), 
                          interpolation=str_to_pil_interp('bicubic')),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    image_path = './base.jpg'
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    print("Initializing TeacherMPLUG model...")
    model = TeacherMPLUG()
    model = model.to(device)
    model.eval()
    
    print("Performing inference...")
    with torch.no_grad():
        topk_ids, topk_probs = model.beam_search(image, train=False, out_size=5)
        
        for i, (tensor_list, prob_list) in enumerate(zip(topk_ids, topk_probs)):
            prob_value = prob_list[0] if isinstance(prob_list, list) and len(prob_list) > 0 else prob_list
            
            if hasattr(prob_value, 'item'):
                prob_value = prob_value.item()
            
            print(f"\n===== Candidate description {i+1} =====")
            
            for j, tensor in enumerate(tensor_list):
                if isinstance(tensor, torch.Tensor):
                    token_ids = tensor.tolist()
                    generated_caption = model.tokenizer.decode(token_ids, skip_special_tokens=True)
                    print(f"  Sentence {j+1}: {generated_caption}") 
import os
import sys
import logging
import torch
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
import torch.nn as nn

# Add mPLUG directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
mplug_path = os.path.join(current_dir, 'mPLUG')
if mplug_path not in sys.path:
    sys.path.append(mplug_path)

# Import mPLUG related modules
from mPLUG.models.model_caption_mplug import MPLUG
from mPLUG.models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from mPLUG.models.visual_transformers import initialize_clip
from mPLUG.models.predictor import TextGenerator
from transformers import BertTokenizer

class TeacherMPLUG(MPLUG):
    def __init__(self):
        """Initialize TeacherMPLUG model"""
        # First call the initialization method of nn.Module
        nn.Module.__init__(self)
        
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.info("Initializing TeacherMPLUG model")
        
        # Initialize tokenizer
        tokenizer_path = cfg.MODEL.MPLUG.TOKENIZER_PATH if hasattr(cfg.MODEL, 'MPLUG') and hasattr(cfg.MODEL.MPLUG, 'TOKENIZER_PATH') else "bert-base-uncased"
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=False)
            self.logger.info(f"Using tokenizer: {tokenizer_path}")
        except OSError as e:
            self.logger.warning(f"Failed to load tokenizer online: {e}")
            self.logger.info("Attempting to load tokenizer offline...")
            try:
                # Try loading from local path
                local_tokenizer_path = os.path.join(current_dir, 'pretrained_models', 'bert-base-uncased')
                if os.path.exists(local_tokenizer_path):
                    self.tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)
                    self.logger.info(f"Loaded tokenizer from local path: {local_tokenizer_path}")
                else:
                    raise OSError(f"Local path {local_tokenizer_path} does not exist")
            except Exception as e2:
                self.logger.error(f"Failed to load tokenizer: {e2}")
                self.logger.info("Please manually download bert-base-uncased model to pretrained_models directory or ensure network connection is available")
                raise
        
        # Set vocabulary size
        if hasattr(cfg.MODEL, 'TEACHER_VOCAB_SIZE'):
            self.vocab_size = cfg.MODEL.TEACHER_VOCAB_SIZE + 1
        else:
            self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        
        # Build mPLUG configuration
        self.config = self.build_config()
        
        # Initialize model components
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
        
        self.use_checkpoint = config["use_checkpoint"] if "use_checkpoint" in config else True
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
            'clip_model_path': cfg.MODEL.MPLUG.CLIP_MODEL_PATH if hasattr(cfg.MODEL.MPLUG, 'CLIP_MODEL_PATH') else "pretrained_models/vit/ViT-B-16.tar"
        }
        
        if hasattr(cfg.MODEL, 'MPLUG'):
            for key in config:
                if hasattr(cfg.MODEL.MPLUG, key.upper()):
                    config[key] = getattr(cfg.MODEL.MPLUG, key.upper())
        
        # config['image_res'] = 256
        
        self.logger.info(f"mPLUG configuration: {config}")
        self.logger.info(f"Teacher model using image resolution: {config['image_res']}x{config['image_res']}")
        return config
    
    def forward(self, att_feats, seq=None, train=True, out_size=5):
        """Forward propagation function"""
        # Convert image features to correct dtype
        image = att_feats.to(dtype=next(self.parameters()).dtype)
        
        # Get image features and attention mask
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        if train and seq is not None:
            # Training mode - using teacher forcing
            # seq is already in tensor form, no need to convert
            answer = seq
            
            # Create attention_mask (1 for non-padding positions)
            attention_mask = (answer != self.tokenizer.pad_token_id).long()
            
            # Create target tensor, set padding positions to -100
            answer_targets = answer.clone()
            answer_targets[answer == self.tokenizer.pad_token_id] = -100
            
            # Call text_decoder
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
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
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
        if hasattr(cfg.MODEL, 'TEACHER_WEIGHTS') and cfg.MODEL.TEACHER_WEIGHTS:
            self.logger.info(f"Loading pre-trained weights: {cfg.MODEL.TEACHER_WEIGHTS}")
            try:
                # Load weight file
                checkpoint = torch.load(cfg.MODEL.TEACHER_WEIGHTS, map_location='cpu')
                
                # Get state dictionary
                try:
                    state_dict = checkpoint['model']
                except:
                    state_dict = checkpoint.get('module', checkpoint)
                
                # Only load weights for visual encoder and text decoder
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('visual_encoder') or key.startswith('text_decoder'):
                        filtered_state_dict[key] = value
                
                # Load weights
                msg = self.load_state_dict(filtered_state_dict, strict=False)
                
                # Filter weight loading information, only keep visual encoder and text decoder related information
                filtered_msg = {
                    'missing_keys': [k for k in msg.missing_keys if k=='visual_encoder' or k=='text_decoder'],
                    'unexpected_keys': [k for k in msg.unexpected_keys if k=='visual_encoder' or k=='text_decoder']
                }
                self.logger.info(f"Weight loading information: {filtered_msg}")
                self.logger.info("Pre-trained weights loaded successfully")
            
            except Exception as e:
                self.logger.error(f"Error loading weights: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.info("No pre-trained weights specified, using random initialization")

    def verify_weights(self):
        """Verify weights are loaded correctly"""
        # Check if key components are initialized
        components = [
            self.visual_encoder,
            self.text_encoder,
            self.fusion_encoder,
            self.text_decoder
        ]
        
        for component in components:
            if component is None:
                self.logger.error(f"Component {component.__class__.__name__} not properly initialized")
                return False
        
        # Check if weights contain NaN or inf
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.logger.error(f"Parameter {name} contains NaN or inf")
                return False
        
        self.logger.info("Weight verification passed")
        return True
    
    # Add distillation related methods
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
                param_m.data.copy_(param.data)  # Initialize
                param_m.requires_grad = False  # Not updated through gradient
                
    @torch.no_grad()
    def _momentum_update(self):
        """Use momentum to update model"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum) 
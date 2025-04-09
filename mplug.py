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
        
        # Call parent class initialization first
        super().__init__(tokenizer=self.tokenizer, config=self.config)
        
        # Load pretrained weights
        self.load_pretrained_weights()
    
    def build_config(self):
        """Build mPLUG configuration"""
        # Default configuration
        config = {
            'text_encoder': 'bert-base-uncased',
            'text_decoder': 'bert-base-uncased',
            'bert_config': os.path.join(mplug_path, 'configs/config_bert.json'),
            'vision_width': 768,
            'use_checkpoint': True,
            'image_res': 384,
            'patch_size': 16,
            'max_words': 30,
            'max_tokens': 30,
            'beam_size': 5,
            'min_length': 1,
            'vit': 'base',
            'clip_name': 'ViT-B-16',  # Add clip_name parameter to ensure consistency with original model
            'distill': True  # Enable distillation mode
        }
        
        # If MPLUG configuration exists in config file, use those values
        if hasattr(cfg.MODEL, 'MPLUG'):
            for key in config:
                if hasattr(cfg.MODEL.MPLUG, key.upper()):
                    config[key] = getattr(cfg.MODEL.MPLUG, key.upper())
        
        self.logger.info(f"mPLUG configuration: {config}")
        return config
    
    def forward(self, att_feats, seq=None, train=True, out_size=5):
        """Forward propagation function
        
        Args:
            att_feats: Image features
            seq: Text sequence
            train: Whether in training mode
            out_size: Number of generated results
            
        Returns:
            In training mode returns logits, in inference mode returns generated sequences and probabilities
        """
        # Convert image features to correct dtype
        image = att_feats.to(dtype=next(self.parameters()).dtype)
        
        # Get image features and attention mask
        image_embeds = self.visual_encoder(image)
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
    
    def load_pretrained_weights(self):
        """Load pretrained weights"""
        if hasattr(cfg.MODEL, 'TEACHER_WEIGHTS') and cfg.MODEL.TEACHER_WEIGHTS:
            self.logger.info(f"Loading pretrained weights: {cfg.MODEL.TEACHER_WEIGHTS}")
            try:
                # Load weight file
                weights = torch.load(cfg.MODEL.TEACHER_WEIGHTS, map_location='cpu')
                
                # Load weights for each model component
                # Handle visual_encoder - not using from_pretrained method
                if hasattr(weights, 'visual_encoder') and self.visual_encoder is not None:
                    self.visual_encoder.load_state_dict(weights.visual_encoder)
                elif 'visual_encoder' in weights and self.visual_encoder is not None:
                    self.visual_encoder.load_state_dict(weights['visual_encoder'])
                else:
                    self.logger.warning("visual_encoder weights not found")
                
                # Handle other components
                for component_name in ['text_encoder', 'fusion_encoder', 'text_decoder']:
                    component = getattr(self, component_name, None)
                    if component is not None:
                        if hasattr(weights, component_name):
                            component.load_state_dict(weights[component_name])
                        elif component_name in weights:
                            component.load_state_dict(weights[component_name])
                        else:
                            self.logger.warning(f"{component_name} weights not found")
                
                self.logger.info("Successfully loaded pretrained weights")
                
                # Verify weights are loaded correctly
                self.verify_weights()
            except Exception as e:
                self.logger.error(f"Error loading weights: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.info("No pretrained weights specified, using random initialization")
            
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
    
    def beam_search(self, image, answer=None, train=True, out_size=5):
        """Perform beam search generation, maintaining compatibility with original MPLUG"""
        image_embeds = self.visual_encoder(image)
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
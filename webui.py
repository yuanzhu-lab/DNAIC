from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp
from PIL import Image
from lib.config import cfg, cfg_from_file

import os
import torch
import time
import argparse
import gradio as gr
import lib.utils as utils
import cv2
import models
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "./experiments/snapshot/student_model_19.pth"  # Path to student model weights
CONFIG_PATH = "./experiments/config.yml"  # Path to configuration file
VOCAB_PATH = "./mscoco/txt/coco_vocabulary.txt"  # Path to vocabulary file
EXAMPLE_IMAGE_PATH = "./base.jpg"  # Path to example image

class ImageCaptionGenerator:
    def __init__(self, model_path, vocab_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.vocab = utils.load_vocab(vocab_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=str_to_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )
        
        self.model = models.create(cfg.MODEL.STUDENT_TYPE)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
        
        self.model.eval()
    
    def preprocess_image(self, image_path):
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img = image_path
        
        att_feats = self.transform(img)
        return att_feats.unsqueeze(0)
    
    def generate_caption(self, image_path):
        with torch.no_grad():
            att_feats = self.preprocess_image(image_path).to(self.device)
            
            att_feats = self.model.module.inference_pre(att_feats)
            out = self.model.module.inference(att_feats)
            prob, seq = torch.max(out, -1)
            
            sents = utils.decode_sequence(self.vocab, seq.data)
            sents = utils.clip_chongfu(sents)
            
            return sents[0] if sents else "Unable to generate description"

def caption_image(image):
    start_time = time.time()
    caption = generator.generate_caption(image)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    return caption, f"Processing time: {elapsed_time:.2f} ms"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Captioning Web UI')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to configuration file')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model weights file')
    parser.add_argument('--vocab', type=str, default=VOCAB_PATH, help='Path to vocabulary file')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        cfg_from_file(args.config)
        print(f"Configuration loaded from: {args.config}")
    else:
        print(f"Warning: Configuration file does not exist: {args.config}")
    
    generator = ImageCaptionGenerator(args.model, args.vocab)
    
    interface = gr.Interface(
        fn=caption_image,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Textbox(label="Generated Description"),
            gr.Textbox(label="Processing Time")
        ],
        title="Image Caption Generation",
        description="Upload an image or use the example image to generate descriptive text",
        examples=[EXAMPLE_IMAGE_PATH] if os.path.exists(EXAMPLE_IMAGE_PATH) else None
    )
    
    interface.launch(share=True) 
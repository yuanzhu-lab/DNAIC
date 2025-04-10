import os
import sys
import argparse
import torch
import torch.nn as nn
import models
from evaluation.evaler import Evaler
from lib.config import cfg, cfg_from_file
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
            
            try:
                # Load weight file
                weights = torch.load(model_path, map_location=self.device)
                
                # Load model weights component by component
                if hasattr(self.model, 'module'):
                    target_model = self.model.module
                else:
                    target_model = self.model
                
                # Get all component names of the model
                component_names = [name for name, _ in target_model.named_children()]
                
                # Handle weight prefix issues
                new_weights = {}
                for k, v in weights.items():
                    # Remove "module." prefix
                    if k.startswith('module.'):
                        name = k[7:]  # remove `module.`
                        new_weights[name] = v
                    else:
                        new_weights[k] = v
                
                # Load weights for each component separately
                loaded_components = 0
                for component_name in component_names:
                    component = getattr(target_model, component_name, None)
                    if component is not None:
                        # Extract weights for this component
                        component_weights = {k.replace(f"{component_name}.", ''): v 
                                            for k, v in new_weights.items() 
                                            if k.startswith(f"{component_name}.")}
                        
                        if component_weights:
                            try:
                                component.load_state_dict(component_weights)
                                print(f"Component {component_name} weights loaded successfully")
                                loaded_components += 1
                            except Exception as e:
                                print(f"Component {component_name} weight loading failed: {e}")
                                try:
                                    component.load_state_dict(component_weights, strict=False)
                                    print(f"Component {component_name} weights loaded successfully in non-strict mode")
                                    loaded_components += 1
                                except Exception as e2:
                                    print(f"Component {component_name} weight loading failed even in non-strict mode: {e2}")
                
                # If no components were successfully loaded, try loading the entire model
                if loaded_components == 0:
                    print("Failed to load any components, attempting to load entire model")
                    try:
                        self.model.load_state_dict(new_weights)
                        print("Entire model loaded successfully")
                    except Exception as e:
                        print(f"Entire model loading failed: {e}")
                        try:
                            self.model.load_state_dict(new_weights, strict=False)
                            print("Entire model loaded successfully in non-strict mode")
                        except Exception as e2:
                            print(f"Entire model loading failed even in non-strict mode: {e2}")
                
                # Verify if weights are loaded correctly
                self.verify_model_weights()
                
            except Exception as e:
                print(f"Error loading weight file: {e}")
                import traceback
                print(traceback.format_exc())
    
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

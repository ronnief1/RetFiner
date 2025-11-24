import numpy as np
from transforms import to_min_max_norm_tensor, to_255_norm_tensor
from multimae.parts.input_adapters import PatchedInputAdapter
from multimae.parts.output_adapters import SpatialOutputAdapter
from multimae.parts.multimae_module import MultiMAE
from functools import partial
import torch
#from peft import LoraConfig, get_peft_model
from timm.models.layers import trunc_normal_

def image_loading(scan):
    full_scan = np.load('../../../jmoran82/FullVIBES-v2/bscan/' + scan + '.npy')#, allow_pickle=True, mmap_mode='r')

    image = to_min_max_norm_tensor(full_scan)

    return image    

def image_loading_255(scan):
    full_scan = np.load('../../../jmoran82/FullVIBES-v2/bscan/' + scan + '.npy')#, allow_pickle=True, mmap_mode='r')

    image = to_255_norm_tensor(full_scan)

    return image  

def image_no_norm(scan):
    full_scan = np.load('../../../jmoran82/FullVIBES-v2/bscan/' + scan + '.npy')#, allow_pickle=True, mmap_mode='r')

    return torch.tensor(full_scan).unsqueeze(0).float()

def slo_loading(scan):
    full_scan = np.load('../../../jmoran82/FullVIBES-v2/slo/' + scan + '.npy')#, allow_pickle=True, mmap_mode='r')

    image = to_min_max_norm_tensor(full_scan)

    return image    


DOMAIN_CONF = {
    'bscan': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    }
}

def get_model(args):
    """Creates and returns model from arguments
    """
    print(
        f"Creating model: {args.model} for inputs {args.in_domains}"
        f" and outputs {args.out_domains}"
    )
    all_domains = set(args.in_domains + args.out_domains)
    if isinstance(args.patch_size, int):
        args.patch_size = {
            domain: (args.patch_size, args.patch_size)
            for domain in all_domains
        }

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            image_size=args.input_size[domain],
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn,
            image_size=args.input_size[domain],
        )
        for domain in args.out_domains
    }

    if 'large' in args.model or '-l' in args.model:
        dim_tokens = 1024
        depth = 24
        num_heads = 16
    else:
        dim_tokens = 768
        depth = 12
        num_heads = 8

    model = MultiMAE(
        args=args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
        dim_tokens=dim_tokens,
        depth=depth,
        num_heads=num_heads,
    )

    return model


def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75, num_layers=None
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if num_layers is None:
        #import pdb; pdb.set_trace()
        num_layers = len(model.multimae.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


def load_vision_backbone(vision_weights_path):

    if 'mirage' in vision_weights_path.lower():
        from mirage_hf import MIRAGEWrapper
        from huggingface_hub import PyTorchModelHubMixin
        
        if 'large' in vision_weights_path.lower():
            print('Loading MIRAGE-Large')
            class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
                def __init__(
                    self,
                    input_size=512,
                    patch_size=32,
                    modalities='bscan',
                    size='large',
                ):
                    super().__init__(
                        input_size=input_size,
                        patch_size=patch_size,
                        modalities=modalities,
                        size=size,
                    )
            vision_encoder = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Large")
            
            del vision_encoder.model.input_adapters.slo

            vision_encoder.proj = torch.nn.Linear(1024, 768)
            trunc_normal_(vision_encoder.proj.weight, std=2e-5)

            def new_forward(*args):
                x = vision_encoder.forward(*args)
                return vision_encoder.proj(x)
                
            vision_encoder.forward_features = new_forward
            vision_encoder.get_vision_features = vision_encoder.forward_features
        
        elif 'base' in vision_weights_path.lower():
            print('Loading MIRAGE-Base')
            class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
                def __init__(
                    self,
                    input_size=512,
                    patch_size=32,
                    modalities='bscan',
                    size='base',
                ):
                    super().__init__(
                        input_size=input_size,
                        patch_size=patch_size,
                        modalities=modalities,
                        size=size,
                    )
            vision_encoder = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Base")
            del vision_encoder.model.input_adapters.slo

            vision_encoder.get_vision_features = vision_encoder.forward
            
        else:
            print('Could not find MIRAGE')
        non_trainable_params = [(name, p.shape) for name, p in vision_encoder.named_parameters() if not p.requires_grad]

        print("Non-trainable parameters:")
        for name, shape in non_trainable_params:
            print(f"{name}: {shape}")
        # print('Loading new mirage implementation')
        # from MultiMAEWrapper import MultiMAEWrapper
        # from mirage_hf import MIRAGEWrapper
        # from huggingface_hub import PyTorchModelHubMixin

        # if 'base' in vision_weights_path.lower():
        #     print('Using base')
        #     vision_encoder = MultiMAEWrapper(weights='../../v3_multimae-b_pret-multimae_49_1600e_bscan_512-32_checkpoint-1599.pth')
        #     class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
        #         def __init__(
        #             self,
        #             input_size=512,
        #             patch_size=32,
        #             modalities='bscan',
        #             size='base',
        #         ):
        #             super().__init__(
        #                 input_size=input_size,
        #                 patch_size=patch_size,
        #                 modalities=modalities,
        #                 size=size,
        #             )
        #     mirage = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Base")
        #     del mirage.model.input_adapters.slo
        #     vision_encoder.load_state_dict(mirage.state_dict())
        #     vision_encoder.get_vision_features = vision_encoder.forward
        # elif 'large' in vision_weights_path.lower():
        #     print('Using large')
        #     vision_encoder = MultiMAEWrapper(weights='../../v3_multimae-b_pret-multimae_49_1600e_bscan_512-32_checkpoint-1599.pth')
        #     class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
        #         def __init__(
        #             self,
        #             input_size=512,
        #             patch_size=32,
        #             modalities='bscan',
        #             size='large',
        #         ):
        #             super().__init__(
        #                 input_size=input_size,
        #                 patch_size=patch_size,
        #                 modalities=modalities,
        #                 size=size,
        #             )
        #     mirage = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Large")
        #     del mirage.model.input_adapters.slo
        #     vision_encoder.load_state_dict(mirage.state_dict())
        #     vision_encoder.get_vision_features = vision_encoder.forward

    
    elif 'multimae' in vision_weights_path.lower():
        print('Loading MultiMAE weights')
        from MultiMAEWrapper import MultiMAEWrapper
        vision_encoder = MultiMAEWrapper(weights=vision_weights_path)
        vision_encoder.get_vision_features = vision_encoder.forward

    elif 'uni4eye' in vision_weights_path.lower():
        print('Loading Uni4Eye++ weights')
        from vit import vit_large_patch16
        vision_encoder = vit_large_patch16(
            img_size=224,
            num_classes=2,
            drop_path_rate=0.1,
            global_pool=False,
        )

        print('loading weights from ', vision_weights_path) 
        checkpoint = torch.load(vision_weights_path, map_location="cpu")

        checkpoint_model = checkpoint["model"]
        
        state_dict = vision_encoder.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        msg = vision_encoder.load_state_dict(checkpoint_model, strict=False)    
        vision_encoder.get_vision_features = vision_encoder.forward_features

    elif 'retfound' in vision_weights_path.lower():
        print('Loading RETFound weights')
        from vit import vit_large_patch16
        vision_encoder = vit_large_patch16(
            img_size=224,
            num_classes=2,
            drop_path_rate=0.1,
            global_pool=False,
        )

        print('loading weights from ', vision_weights_path) 
        checkpoint = torch.load(vision_weights_path, map_location="cpu")
        
        checkpoint_model = checkpoint["model"]
        
        state_dict = vision_encoder.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        msg = vision_encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        vision_encoder.get_vision_features = vision_encoder.forward_features

    elif 'urfound' in vision_weights_path.lower():
        print('Loading UrFound weights')
        from vit import vit_base_patch16
        vision_encoder = vit_base_patch16(
            img_size=224,
            num_classes=2,
            drop_path_rate=0.1,
            global_pool=False,
            use_proj=False
        )

        print('loading weights from ', vision_weights_path) 
        checkpoint = torch.load(vision_weights_path, map_location="cpu")
        
        checkpoint_model = checkpoint["model"]
        
        state_dict = vision_encoder.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = vision_encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        vision_encoder.get_vision_features = vision_encoder.forward_features
    
    elif 'visionfm' in vision_weights_path.lower():
        print('Loading VisionFM weights')
        from VisionFM_main.models.vision_transformer import VisionTransformer
        vision_encoder = VisionTransformer(return_all_tokens=True, qkv_bias=True)
        print('loading weights from ', vision_weights_path) 
        state_dict = torch.load(vision_weights_path, map_location='cpu')['teacher']
        for key in list(state_dict.keys()):
            if 'backbone.' in key:
                state_dict[key.replace('backbone.', '')] = state_dict.pop(key)

        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        vision_encoder.get_vision_features = vision_encoder.forward

    # elif 'clip' in vision_weights_path:
    #     from transformers import CLIPModel
    #     vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model

    else:
        raise ValueError("Undefined vision backbone. Don't know how to load. Add your vision model loading code to util_funcs.py in the function load_vision_backbone.")

    return vision_encoder

def get_model_name(model_weights):
    if 'multimae' in model_weights:
        return 'MAE'
    elif 'retfound' in model_weights:
        return 'RETFound'
    elif 'visionfm' in model_weights:
        return 'VisionFM'
    elif 'uni4eye' in model_weights:
        return 'Uni4Eye++'
    elif 'mirage' in model_weights:
        if 'large' in model_weights:
            return 'MIRAGE-Large'
        else:
            return 'MIRAGE-Base'
    elif 'urfound' in model_weights:
        return 'UrFound'
    else:
        raise ValueError("Undefined vision backbone. Don't know how to load. Add your vision model loading code to util_funcs.py in the function load_vision_backbone.")
        

def get_data_loaders(model_weights, args):
    from pandas import read_csv
    from ImageCaptionDataset import ImageCaptionDataset
    from torch.utils.data import DataLoader
    from transforms import collate_fn, qwen_collate_fn

    train_data = read_csv('vibes_data/no_test/train.csv', index_col=0).reset_index(drop=True)
    val_data = read_csv('vibes_data/no_test/val.csv', index_col=0).reset_index(drop=True)


    if 'multimae' in model_weights or 'mirage' in model_weights:        
        train_dataset = ImageCaptionDataset(train_data)
        val_dataset = ImageCaptionDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=10) #Define your own dataloader
    
    elif 'retfound' in model_weights or 'uni4eye' in model_weights:
        from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
        transform = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Copy the single channel to 3 channels
            Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet means
                               std=[0.229, 0.224, 0.225])
        ])

        train_dataset = ImageCaptionDataset(train_data, transform, load_255=True)
        val_dataset = ImageCaptionDataset(val_data, transform, load_255=True)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=10) #Define your own dataloader

    elif 'urfound' in model_weights:
        from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
        transform = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Copy the single channel to 3 channels
            Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet means
                               std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageCaptionDataset(train_data, transform, load_255=True)
        val_dataset = ImageCaptionDataset(val_data, transform, load_255=True)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=10) #Define your own dataloader

    
    elif 'visionfm' in model_weights:
        from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC

        transform = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Copy the single channel to 3 channels
            # Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet means
            #                    std=[0.229, 0.224, 0.225])
            Normalize(mean = [0.21091926, 0.21091926, 0.21091919],
                std = [0.17598894, 0.17598891, 0.17598893]) 
        ])
        train_dataset = ImageCaptionDataset(train_data, transform, load_255=True)
        val_dataset = ImageCaptionDataset(val_data, transform, load_255=True)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=10) #Define your own dataloader

    elif 'qwen' in model_weights:
        from QwenImageCaptionDataset import QwenImageCaptionDataset
        train_dataset = QwenImageCaptionDataset(train_data)
        val_dataset = QwenImageCaptionDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=qwen_collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=qwen_collate_fn, shuffle=False, num_workers=10) #Define your own dataloader

    else:
        raise ValueError("Undefined vision backbone. Don't know how to load. Add your vision model loading code to util_funcs.py in the function load_vision_backbone.")


    return train_loader, val_loader

def get_losses_info(args):
    if args.losses is not None:
        loss_dict = {loss: 1 if loss in args.losses else 0 for loss in ["mlm", "gm", "itm", "itc"]}
        loss_names = ''.join(args.losses).upper()
    else:
        loss_names = 'LossesAll'
        loss_dict = {loss: 1 for loss in ["mlm", "gm", "itm", "itc"]}
    
    return loss_names, loss_dict

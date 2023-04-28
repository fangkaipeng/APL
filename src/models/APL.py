import sys
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()
from models.PromptedCLIP import TextEncoder


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP, device):
        super().__init__()
        n_cls = len(classnames) 
        n_ctx = cfg.tp_N_CTX  
        dtype = clip_model.dtype 
        self.ctx_dim = clip_model.ln_final.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution 
        cfg_imsize = cfg.image_size 
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=dtype).to(device) 
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        prompts = ["a photo of " + name + " from " + "X "*n_ctx +"domain." for name in classnames] 
        self.prefix_index = [length+5 for length in name_lens] 
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("origin_text_embedding",embedding)
      
        self.tokenized_prompts = tokenized_prompts 
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 
        
        prompts = [torch.cat([self.origin_text_embedding[i,:self.prefix_index[i]],ctx[i],self.origin_text_embedding[i,self.prefix_index[i]+self.n_ctx:]],dim=0).view(1,-1,self.ctx_dim) for i in range(self.n_cls)]
        prompts = torch.cat(prompts, dim=0)
        return prompts


class APL(nn.Module):
    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
       
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        self.cls_num_tokens = len(self.dict_clss)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        self.generator = copy.deepcopy(clip.visual.transformer)
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        if self.cfg.DOM_PROJECT > -1:
            # only for prepend / add
            sp_dom_prompt_dim = self.cfg.DOM_PROJECT
            self.sp_dom_prompt_proj = nn.Linear(
                sp_dom_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_dom_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_dom_prompt_dim = width
            self.sp_dom_prompt_proj = nn.Identity()  # 占位

        if self.cfg.CLS_PROJECT > -1:
            # only for prepend / add
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(
                sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()  # 占位

        # definition of specific prompts 
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa
        self.specific_domain_prompts = nn.Parameter(torch.zeros(self.dom_num_tokens, sp_dom_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
        self.specific_class_prompts = nn.Parameter(torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)
        
        # definition of Parameter about generated prompts 
        scale = width ** -0.5
        self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
        self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))

    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
        x = x.permute(0, 2, 1) # 65 49 768
        base = self.cfg.GP_CLS_NUM_TOKENS+self.cfg.GP_DOM_NUM_TOKENS + 1
        if stage == 1:
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts[dom_index]).view(B, 1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts[cls_index]).view(B, 1, -1), 
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            
            sp_dom_prompts = self.sp_dom_prompt_proj(self.specific_domain_prompts)
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)
    
            cls_prompt_mask = torch.zeros(B, sp_cls_prompts.shape[0], sp_cls_prompts.shape[1]).type(torch.bool).to(self.device)
            dom_prompt_mask = torch.zeros(B, sp_dom_prompts.shape[0], sp_dom_prompts.shape[1]).type(torch.bool).to(self.device)
            cls_prompt_mask[range(B), cls_index, :] = 1
            dom_prompt_mask[range(B), dom_index, :] = 1
        
            sp_cls_prompts = sp_cls_prompts.expand(B,-1,-1).masked_fill(cls_prompt_mask, 0)
            sp_dom_prompts = sp_dom_prompts.expand(B,-1,-1).masked_fill(dom_prompt_mask, 0) 

            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1,-1),
                self.ge_cls_prompt_template.expand(B,-1, -1),
                sp_dom_prompts,
                sp_cls_prompts,
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
    
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            dom_prompts.view(B, self.cfg.GP_DOM_NUM_TOKENS, -1),
            cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1, -1),
                self.ge_cls_prompt_template.expand(B,-1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B,-1,-1),
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1,-1),
                x + self.clip_positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        if out_token == 1:
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,:out_token,:])
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id,3, x[:,:self.cfg.GP_DOM_NUM_TOKENS], x[:,self.cfg.GP_DOM_NUM_TOKENS:]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, x[:,:self.cfg.GP_DOM_NUM_TOKENS], x[:,self.cfg.GP_DOM_NUM_TOKENS:])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)
    

    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_doms = dict_doms
        self.dict_clss = dict_clss
        self.cls_num_tokens = len(self.dict_clss)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        self.generator = copy.deepcopy(clip.visual.transformer)
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        

        if self.cfg.CLS_PROJECT > -1:
            # only for prepend / add
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(
                sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()  # 占位

        # definition of specific prompts 

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
        self.specific_class_prompts = nn.Parameter(torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)
        
        # definition of Parameter about generated prompts 
        scale = width ** -0.5
        self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
    
    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
        x = x.permute(0, 2, 1) # 65 49 768
      
        if stage == 1:
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts[cls_index]).view(B, 1, -1), 
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            
           
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)
    
            cls_prompt_mask = torch.zeros(B, sp_cls_prompts.shape[0], sp_cls_prompts.shape[1]).type(torch.bool).to(self.device)
            cls_prompt_mask[range(B), cls_index, :] = 1
            
            sp_cls_prompts = sp_cls_prompts.expand(B,-1,-1).masked_fill(cls_prompt_mask, 0)
            
            x = torch.cat((
                self.ge_cls_prompt_template.expand(B,-1, -1),
                sp_cls_prompts,
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding

            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = torch.cat((
                self.ge_cls_prompt_template.expand(B,-1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1,-1),
                x + self.clip_positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        if out_token == 1:
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,:out_token,:])
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_CLS_NUM_TOKENS+1) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id,3, cls_prompts=x[:,1]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_CLS_NUM_TOKENS+1)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, cls_prompts=x[:,1])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)

    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
       
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        self.generator = copy.deepcopy(clip.visual.transformer)
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        if self.cfg.DOM_PROJECT > -1:
            # only for prepend / add
            sp_dom_prompt_dim = self.cfg.DOM_PROJECT
            self.sp_dom_prompt_proj = nn.Linear(
                sp_dom_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_dom_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_dom_prompt_dim = width
            self.sp_dom_prompt_proj = nn.Identity()  # 占位


        # definition of specific prompts 
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa
        self.specific_domain_prompts = nn.Parameter(torch.zeros(self.dom_num_tokens, sp_dom_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

        # definition of Parameter about generated prompts 
        scale = width ** -0.5
        self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))

    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
        x = x.permute(0, 2, 1) # 65 49 768
        if stage == 1:
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts[dom_index]).view(B, 1, -1),
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            
            sp_dom_prompts = self.sp_dom_prompt_proj(self.specific_domain_prompts)
            dom_prompt_mask = torch.zeros(B, sp_dom_prompts.shape[0], sp_dom_prompts.shape[1]).type(torch.bool).to(self.device)
            dom_prompt_mask[range(B), dom_index, :] = 1
        
            sp_dom_prompts = sp_dom_prompts.expand(B,-1,-1).masked_fill(dom_prompt_mask, 0) 

            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1,-1),
                sp_dom_prompts,
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding
        
            
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            dom_prompts.view(B, self.cfg.GP_DOM_NUM_TOKENS, -1),
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B,-1,-1),
                x + self.clip_positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        if out_token == 1:
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,:out_token,:])
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+1) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id,3, dom_prompts=x[:,1]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+1)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, dom_prompts=x[:,1])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)
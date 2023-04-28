from clip.model import CLIP
import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# gm = GPUmanager.GPUManager()
# gpu_index = gm.auto_choice()
# device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames) # 400
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        ctx_dim = clip_model.ln_final.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution 
        cfg_imsize = cfg.image_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device) 
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) 
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 

        prefix = self.token_prefix 
        suffix = self.token_suffix 
        
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts 

class PromptedVisionTransformer(nn.Module):
    def __init__(self, config,model:CLIP):
        super(PromptedVisionTransformer, self).__init__()
        self.input_resolution = model.visual.input_resolution
        self.output_dim = model.visual.output_dim
        self.conv1 = model.visual.conv1
        width = self.conv1.out_channels
        self.class_embedding = model.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.ln_post = model.visual.ln_post
        self.proj = model.visual.proj
        self.config = config
        patch_size = self.conv1.kernel_size
        num_tokens = self.config.vp_NUM_TOKENS  # "10"
        self.num_tokens = num_tokens  # number of prompted tokens
        if self.config.vp_PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.config.vp_PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity() 

        if self.config.vp_INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim)) # layer, num_token, prompt_dim
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.config.vp_DEEP:  # noqa

                total_d_layer = self.transformer.layers - 1 
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        # after CLS token, all before image patches
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype) # [65, 50, 678] + [50 ,768]
        x = torch.cat((
            x[:, :1, :], # CLS token
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1),
            x[:, 1:, :]
        ), dim=1)
        return x

    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = self.transformer.layers
        # print("yes")
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output) 
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

                hidden_states= self.transformer.resblocks[i](hidden_states)
        return hidden_states

    def forward(self, x):
        x = self.incorporate_prompt(x)
        if self.config.vp_DEEP:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.forward_deep_prompt(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        else:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        return x
       
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP, device):
        super().__init__()
        self.cfg = cfg
        if cfg.training_strategy == 'TP' or cfg.training_strategy == 'TP+VP':
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model, device).to(device)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
            self.text_encoder = TextEncoder(clip_model).to(device)
        else :
            self.text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)
            # self.text_features = clip_model.encode_text(self.text_template)
            self.text_encoder = clip_model.encode_text
        if cfg.training_strategy == 'VP' or cfg.training_strategy == 'TP+VP':
            self.image_encoder = PromptedVisionTransformer(cfg, clip_model)
        else :
            self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype 

    def forward(self, image): # bt 3, 244, 244
        image_features = self.image_encoder(image.type(self.dtype)) # batch, 512
        if self.cfg.training_strategy == 'TP' or self.cfg.training_strategy == 'TP+VP':
            prompts = self.prompt_learner() # 400,77,512
            tokenized_prompts = self.tokenized_prompts # 400,77 400类，描述每个类的text token化后为77长度
            text_features = self.text_encoder(prompts, tokenized_prompts) # class, 512
        else :
            text_features = self.text_encoder(self.text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

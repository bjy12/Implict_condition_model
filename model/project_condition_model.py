from typing import Optional, Union

import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin

from torch import Tensor
from einops import rearrange

from utils_file.model_utils import index_2d
from model.modules.image_encoder.u_net import UNet
from model.modules.points_wise.points_encoder import PointsWiseFeatureEncoder
from model.modules.points_wise.position_embedding import get_embedder , GlobalFeatureEncoder
import pdb 

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]


class ProjectionImplictConditionModel(ModelMixin):
    def __init__(
        self,
        image_encoder: dict = {},
        use_coords:bool = True,
        use_local_features: bool = True,
        use_global_features: bool = False,
        encoder_type: str = 'view_mixer' ,
        num_views: int = 2 ,
        gl_f_input_c: int = 321, 
        gl_f_output_c: int = 128,
        merge_mode: str = 'concat'
    ):
        super().__init__()

        self.image_encoder = UNet(**image_encoder)
        self.use_coords = use_coords
        self.use_local_conditioning = use_local_features
        self.use_global_conditioning = use_global_features
        self.num_views = num_views
        self.gl_f_input_c = gl_f_input_c
        self.gl_f_output_c = gl_f_output_c

        self.points_wise_encoder = PointsWiseFeatureEncoder(encoder_type , num_views)
        self.position_embedder , _  = get_embedder(10 , 0 )
        self.gloabl_feature_encoder = GlobalFeatureEncoder(gl_f_input_c , gl_f_output_c , merge_mode )


    def image_wise_encoder(self, projs):
        b , m , c , w , h  = projs.shape
        projs = projs.reshape(b*m, c , w , h)
        proj_feats = self.image_encoder(projs)
        #pdb.set_trace()
        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) # B, M, C, W, H
            #check_feature_range(proj_feats[i] , f"proj_feats_scale_{i}")
        #pdb.set_trace()

        return proj_feats
    
    def get_local_conditioning(self,proj_xray: Tensor , points_proj:Tensor):
        #! keep points order 
        #b , c , h  , w  ,d = points_proj.shape
        #points_proj = rearrange(points_proj, 'b c h w d -> b c (h w d)') 
        #pdb.set_trace()
        b , m , c , w , h = proj_xray.shape
        #pdb.set_trace()
        #proj_xray = proj_xray.reshape(b*m , c, w, h)
        proj_xray = rearrange(proj_xray , "b m c w h -> (b m) c w h")
        xray_feats , global_features = self.image_encoder(proj_xray) # xray feats (b m) c w d   global_features (b m) c 1 1   m is n_view 
        #pdb.set_trace()
        #pdb.set_trace()
        global_features = global_features.squeeze(-1).squeeze(-1)
        global_features = rearrange(global_features , '(b m) c -> b (m c) ', b = b , m = m) # b m c 
        #pdb.set_trace()
        c_out = global_features.shape[1]

        #pdb.set_trace()

        #global_features = global_features.reshape(b,m,c_out,1,1)
        #pdb.set_trace()
        xray_feats  = list(xray_feats) if type(xray_feats) is tuple else [xray_feats]
        for i in range(len(xray_feats)):
            feat = xray_feats[i]
            _, c_, w_, h_ = feat.shape
            xray_feats[i] = rearrange(feat , " ( b m ) c w h -> b m c w h" , b = b , m=m)
            #xray_feats[i] = xray_feats[i].reshape(b, m, c_, w_, h_) # B, M, C, W, H
        #pdb.set_trace()
        points_feats = self.project_points(xray_feats , points_proj)

        # points_wise_feature_process
        points_feats = self.points_wise_encoder(points_feats)
        #pdb.set_trace()
        local_feats = points_feats

        return local_feats , global_features




    def project_points(self, proj_feats ,points_proj):
        n_view = proj_feats[0].shape[1]
        # query view_specific features 
        p_list = []
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                feat = proj_f[:, i, ...] # B, C, W, H
                p = points_proj[:, i, ...] # B, N, 2
                # if torch.any(torch.abs(p) > 1):
                #     print(f"Warning: coordinates out of range [-1,1]: {p.min():.3f} to {p.max():.3f}")
                p_feats = index_2d(feat, p) # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.stack(p_list, dim=-1) # B, C, N, M
        return p_feats


    def get_input_with_conditioning(
        self,
        x_t: Tensor,                    # b, 1 , h , w, d
        projs_xray: Optional[Tensor],    # b, 2 , u , v
        points_proj: Optional[Tensor],  # b, 2 , h , w , d 
        coords: Optional[Tensor],       # b, 3 , h , w , d
    ):
        # 
        #B , C , H , W , D = x_t.shape
        #pdb.set_trace()
        x_t_input = [x_t]
        #x_t_input.append(coords)
        B , H , W , D , _  = coords.shape
        #condtion_list = []
        #pdb.set_trace()
        if self.use_coords:
            x_t_input.append(coords)

        local_features , global_features = self.get_local_conditioning(projs_xray, points_proj)

        if self.use_local_conditioning:
            #pdb.set_trace()
            local_features = rearrange(local_features , 'b c (h w d) -> b h w d c', h=H, w=W, d=D) 
            #pdb.set_trace()
            x_t_input.append(local_features)

            #pdb.set_trace()
        if self.use_global_conditioning:
            #pdb.set_trace()
            global_features = global_features.unsqueeze(-1) # b 1 (n_view * c)
            global_features = global_features.repeat(1,1,(H*W*D))  
            #global_features = rearrange(global_features , 'b c ( h w d ) -> b c h w d ' , h=H , w = W , d = D)
            global_features = rearrange(global_features , ' b c n -> b n c ')

            #pdb.set_trace()

            # global_features = rearrange(global_features , "b n_v c -> b (n_v c) 1 1 1 ").expand(-1,-1,H,W,D)
            # global_features = global_features.permute(0,2,3,4,1) # 64 64 64 258
            # global_features = rearrange(global_features , ' b c h w d -> b h w d c')
            #pdb.set_trace()
            
            global_coords = coords
            global_coords = rearrange(global_coords , "b h w d c -> b ( h w d ) c " , h=H , w=W , d=D)

            #pdb.set_trace()
            embedded_coords = self.position_embedder(global_coords)
            #pdb.set_trace()
            #embedded = rearrange(embedded , "b (h w d) c -> b h w d c" , h=H , w=W , d=D)
            #embedded = embedded.reshape(B, H, W, D, -1)  # 64 64 64  63
            #pdb.set_trace()
            outputs_global = self.gloabl_feature_encoder(embedded_coords, global_features)  # input channels ( 63+258 =  )  output channnels   (128)  
            #pdb.set_trace()
            outputs_global = rearrange(outputs_global , " b (h w d ) c -> b h w d c " , h=H , w=W , d=D)
            x_t_input.append(outputs_global)
            #pdb.set_trace()

        
        x_t_input  = torch.cat(x_t_input , dim=-1)  # (B, h , w ,d , noised_i+pos+lcoal_d+global_d )
        
        x_t_input = rearrange(x_t_input , ' b h w d c -> b c h w d ')
        
        #pdb.set_trace()
          
        return x_t_input
    
    def forward(self, batch: dict, mode: str = 'train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()

        

        
import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.BatchNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)




class ViewMixer(nn.Module):
    def __init__(self, num_views:int = 2):
        super().__init__()
        self.num_views = num_views
        self.view_mixer = MLP([num_views, num_views // 2 , 1])

    def forward(self, p_feats):
        p_feats = p_feats.permute(0, 3, 1, 2)
        p_feats = self.view_mixer(p_feats)
        p_feats = p_feats.squeeze(1)

        return  p_feats



class PointsWiseFeatureEncoder(nn.Module):
    def __init__(self, encoder_type:str = 'view_mixer' , num_views:int = 2):
        super().__init__()

        if  encoder_type == 'view_mixer':
            self.points_wise_encoder = ViewMixer(num_views = num_views)

    def forward(self , p_f):
        p_f = self.points_wise_encoder(p_f)

        return  p_f
    

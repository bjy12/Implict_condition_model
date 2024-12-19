from config.train_cfg_pcc import ProjectConfig
from model.points_wise_diffusion import Points_WiseImplict_ConditionDiffusionModel


def get_model(cfg: ProjectConfig):
    model = Points_WiseImplict_ConditionDiffusionModel(**cfg.model)

    return model
import json
import os
from typing import List, Tuple
from src.models.agent_build_utils import get_env_class_vars

# from src.models.localization.clip_owl import ClipOwl
from src.models.agent_fbe import AgentFbe
import torchvision.transforms as T
from PIL import Image
from src.simulation.constants import (FORWARD_M, FOV,
                                      IMAGE_HEIGHT, IMAGE_WIDTH,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG,
                                      VOXEL_SIZE_M, IN_CSPACE)
from src.simulation.sim_enums import ClassTypes, EnvTypes
from src.simulation.utils import get_device
from torch import device

from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg

from GLEE.app.predictor_glee import GLEEDemo

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AgentFbeGLCLIP(AgentFbe):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            classes: List[str],
            classes_clip: List[str],
            templates: List[str],
            fov: float,
            height: float,
            width: float,
            agent_height: float,
            floor_tolerance: float,
            threshold_glee: float,
            threshold_glip: float,
            device: device,
            max_ceiling_height: float = MAX_CEILING_HEIGHT_M,
            rotation_degrees: int = ROTATION_DEG,
            forward_distance: float = FORWARD_M,
            voxel_size_m: float = VOXEL_SIZE_M,
            in_cspace: bool = IN_CSPACE,
            debug_dir: str = None,
            wandb_log: bool = False,
            negate_action: bool = False,
            fail_stop: bool = True,
            open_clip_checkpoint: str = '',
            alpha: float = 0.,
            center_only: bool = False,
            class_remap=None,
            target_detector: str = 'glclip'
        ):
        super(AgentFbeGLCLIP, self).__init__(fov,
                                               device,
                                               max_ceiling_height=max_ceiling_height,
                                               rotation_degrees=rotation_degrees,
                                               forward_distance=forward_distance,
                                               agent_height=agent_height,
                                               floor_tolerance=floor_tolerance,
                                               voxel_size_m=voxel_size_m,
                                               in_cspace=in_cspace,
                                               debug_dir=debug_dir,
                                               wandb_log=wandb_log,
                                               negate_action=negate_action,
                                               fail_stop=fail_stop,
                                               open_clip_checkpoint=open_clip_checkpoint,
                                               alpha=alpha,
                                               target_detector=target_detector)

        # # init glip model
        # config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
        # weight_file = "GLIP/MODEL/glip_large_model.pth"
        # glip_cfg.local_rank = 0
        # glip_cfg.num_gpus = 2  # 1
        # glip_cfg.merge_from_file(config_file)
        # glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        # glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        # self.glip_module = GLIPDemo(
        #     glip_cfg,
        #     min_image_size=800,
        #     confidence_threshold=threshold_glip,
        #     show_mask_heatmaps=False,
        #     device=device,
        #     classes_clip=classes_clip,
        # )
        #
        # # init glee model
        # self.glee_module = GLEEDemo(
        #     model_selection='GLEE-Plus (SwinL)',
        #     device=device,
        #     confidence_threshold=threshold_glee,
        #     classes=classes,
        #     classes_clip=classes_clip,
        # )


def build(fail_stop, prompts_path, threshold_glee, threshold_glip, open_clip_checkpoint='', alpha=0., clip_model_name="ViT-B/32",
          device_num=-1, debug_dir=None, wandb_log=False, env_type=EnvTypes.ROBOTHOR, class_type=ClassTypes.REGULAR,
          center_only=False, class_remap=None):
    classes, classes_clip, agent_height, floor_tolerance, negate_action, prompts = get_env_class_vars(prompts_path,
                                                                                                      env_type,
                                                                                                      class_type)

    agent_class = AgentFbeGLCLIP
    agent_kwargs = {
        "classes": classes,
        "classes_clip": classes_clip,
        "templates": prompts,
        "fov": FOV,
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "agent_height": agent_height,
        "floor_tolerance": floor_tolerance,
        "device": get_device(device_num),
        "debug_dir": debug_dir,
        "wandb_log": wandb_log,
        "negate_action": negate_action,
        "fail_stop": fail_stop,
        "open_clip_checkpoint": open_clip_checkpoint,
        "alpha": alpha,
        "threshold_glee": threshold_glee,
        "threshold_glip": threshold_glip,
        "center_only": center_only,
        "class_remap": class_remap,
    }

    render_depth = True

    return agent_class, agent_kwargs, render_depth

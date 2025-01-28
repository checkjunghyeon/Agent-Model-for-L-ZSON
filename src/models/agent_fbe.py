import copy
import os
from typing import List, Tuple
from abc import ABC, abstractmethod

# from skimage import filters
from src.models.agent import Agent
from src.models.exploration.frontier_based_exploration import FrontierBasedExploration
from src.models.exploration.frontier_psl_based_exploration import FrontierPSLBasedExploration
from src.simulation.constants import (FORWARD_M,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG, VOXEL_SIZE_M, IN_CSPACE,
                                      THOR_LANDMARK_TYPES, THOR_ROOM_TYPES,
                                      THOR_OBJECT_TYPES_MAP, THOR_LONGTAIL_TYPES_MAP)
from torch import device, is_tensor
from src.models.agent_mode import AgentMode
from threadpoolctl import threadpool_limits
import torch
import numpy as np
import matplotlib.pyplot as plt

PREFIX = os.path.dirname(os.path.abspath(__file__)) + "/"
OUTPUT_PATH = PREFIX + "images_output_glip/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def localize_(box):
    image_relevance = torch.zeros((672, 672))

    box = [int(round(i, 2)) for i in box.tolist()]
    image_relevance[box[1]:box[3], box[0]:box[2]] = 1.

    return image_relevance


def localize(boxes, labels, scores, type_=None):
    image_relevance = torch.zeros((len(type_), 672, 672))

    for j, label in enumerate(labels):
        if label in type_:
            score = scores[j]
            bbox = boxes[j]

            box = [int(round(i, 2)) for i in bbox.tolist()]
            image_relevance[type_.index(label), box[1]:box[3], box[0]:box[2]] = score

    return image_relevance


class AgentFbe(Agent, ABC):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            fov: float,
            device: device,
            agent_height: float,
            floor_tolerance: float,
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
            alpha: float = 0.):

        super(AgentFbe, self).__init__()

        temp_flag = "PSL"

        if temp_flag == "PSL":
            self.fbe = FrontierPSLBasedExploration(fov, device, max_ceiling_height, agent_height,
                                                   floor_tolerance, rotation_degrees, forward_distance,
                                                   voxel_size_m, in_cspace, wandb_log, negate_action, fail_stop)
        else:
            self.fbe = FrontierBasedExploration(fov, device, max_ceiling_height, agent_height,
                                                floor_tolerance, rotation_degrees, forward_distance,
                                                voxel_size_m, in_cspace, wandb_log, negate_action, fail_stop)
        self.timesteps = 0
        self.debug_dir = debug_dir
        self.debug_data = []
        if debug_dir is not None:
            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)

        self.agent_mode = AgentMode.SPIN

        self.action_queue = []

        self.rotation_degrees = rotation_degrees
        self.forward_distance = forward_distance
        assert (360 - int(fov)) % self.rotation_degrees == 0
        self.rotation_counter = 0
        self.max_rotation_count = (360 - int(fov)) / self.rotation_degrees
        self.last_action = None
        self.open_clip_checkpoint = open_clip_checkpoint
        self.alpha = alpha

        self.glip_predictions = []
        self.land_predictions = []
        self.room_predictions = []
        self.reverse_thor_obj_types_map = {v: k for k, v in THOR_OBJECT_TYPES_MAP.items()}
        self.reverse_thor_longtail_types_map = {v: k for k, v in THOR_LONGTAIL_TYPES_MAP.items()}

    def reset(self):
        self.timesteps = 0
        self.fbe.reset()
        self.agent_mode = AgentMode.SPIN
        self.last_action = None

    def act(self, observations):
        # analyse observation for object
        with threadpool_limits(limits=1):
            attention = self.localize_object(observations)
            land_attention = self.localize_landmarks(observations)
            room_attention = self.localize_rooms(observations)

            self.debug_data.append((self.timesteps, attention.max().item()))

            # with threadpool_limits(limits=1):
            # update map
            self.fbe.update_map(
                observations,
                self.glee_module.glee_caption[observations['object_goal']],
                attention,
                land_attention,  # added
                room_attention,  # added
                self.last_action,
                self.agent_mode  # added
            )

            if self.fbe.poll_roi_exists() and self.agent_mode != AgentMode.EXPLOIT:
                self.rotation_counter = 0
                self.action_queue = []

                # NOTE: uncomment for fig
                self.agent_mode = AgentMode.EXPLOIT

            elif self.agent_mode == AgentMode.SPIN and self.rotation_counter == self.max_rotation_count:
                #알려지지 않은 영역을 탐험
                self.rotation_counter = 0
                self.action_queue = []
                self.agent_mode = AgentMode.EXPLORE

            # determine action to take
            action = None

            if self.agent_mode == AgentMode.SPIN:
                action = self.rotate()
            elif self.agent_mode == AgentMode.EXPLORE:
                action = self.explore(observations, attention, land_attention, room_attention)
            elif self.agent_mode == AgentMode.EXPLOIT:
                action = self.exploit(observations, attention, land_attention, room_attention)

            self.timesteps += 1
            self.last_action = action
        print(f'timestamp : {self.timesteps} \t action : {action}')
        return action


    def localize_object(self, observations):
        rgb = observations["rgb"][:, :, [2, 1, 0]]

        boxes = self.glee_module.demo(input_img=rgb,
                                      goal=observations["object_goal"],
                                      save=False)

        if boxes.size(0) == 0:
            print("# No Predictions by GLEE")
            return torch.zeros((672, 672))

        detected_img, self.glip_predictions = self.glip_module.inference(
            original_image=rgb,
            target="goal",
            thresh=0.58)

        new_labels = self.get_glip_real_label(self.glip_predictions)
        self.glip_predictions.add_field("labels", new_labels)
        goal_labels = self.glip_predictions.get_field("labels")

        iou = 0.
        real_target = observations["object_goal"]

        for j, label in enumerate(goal_labels):
            # print(boxes[0].tolist(), self.glip_predictions.bbox[j].tolist())

            # Uncommon
            if 'crate' in self.glip_module.goal_caption:
                print(f'REAL: {real_target} <=> OBS: {self.reverse_thor_longtail_types_map[label]}')
                if real_target == self.reverse_thor_longtail_types_map[label]:
                    iou = self.calculate_iou(boxes[0].tolist(), self.glip_predictions.bbox[j].tolist())
            # Common(Spatial, Appearance, Hidden)
            elif 'vase' in self.glip_module.goal_caption:
                print(f'REAL: {real_target} <=> OBS: {self.reverse_thor_obj_types_map[label]}')
                if real_target == self.reverse_thor_obj_types_map[label]:
                    iou = self.calculate_iou(boxes[0].tolist(), self.glip_predictions.bbox[j].tolist())
            print("IoU:", iou)
            # self.imshow(detected_img, real_target, real_target + "_" + label + "_" + str(iou))
            if iou > 0.35:
                img = localize_(self.glip_predictions.bbox[j])
                return img

        return torch.zeros((672, 672))

    def localize_landmarks(self, observations):
        _, self.land_predictions = self.glip_module.inference(original_image=observations["rgb"][:, :, [2, 1, 0]],
                                                              target="landmark",
                                                              thresh=0.61)
        new_labels = self.get_glip_real_label(self.land_predictions)
        self.land_predictions.add_field("labels", new_labels)
        land_prediction = copy.deepcopy(self.land_predictions)
        land_labels = land_prediction.get_field("labels")

        print("LANDMARKS:", land_labels)
        return localize(self.land_predictions.bbox,
                        land_labels,
                        self.land_predictions.get_field("scores"),
                        type_=THOR_LANDMARK_TYPES)

    def localize_rooms(self, observations):
        _, self.room_predictions = self.glip_module.inference(observations["rgb"][:, :, [2, 1, 0]], target="room")

        new_labels = self.get_glip_real_label(self.room_predictions)
        self.room_predictions.add_field("labels", new_labels)
        room_labels = self.room_predictions.get_field("labels")

        print("ROOMS:", room_labels)
        return localize(self.room_predictions.bbox,
                        room_labels,
                        self.room_predictions.get_field("scores"),
                        type_=THOR_ROOM_TYPES)

    def get_glip_real_label(self, prediction):
        print(self.glip_module.entities)
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_module.entities and self.glip_module.plus:
            for i in labels:
                if i <= len(self.glip_module.entities):
                    new_labels.append(self.glip_module.entities[i - self.glip_module.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels

    def calculate_iou(self, box_a, box_b):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        box_a : list or tuple
            The (x1, y1, x2, y2) coordinates of the first bounding box.
        box_b : list or tuple
            The (x1, y1, x2, y2) coordinates of the second bounding box.

        Returns
        -------
        float
            The IoU of the two bounding boxes.
        """
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        # Calculate the area of intersection rectangle
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        # Calculate the area of both bounding boxes
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # Calculate the union area by using the formula: union(A,B) = A + B - intersection(A,B)
        union_area = box_a_area + box_b_area - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area

        return iou

    def imshow(self, img, caption, imgname):
        plt.imshow(img[:, :, :])  # [2, 1, 0]
        plt.axis("off")
        plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
        output_image_path = os.path.join(OUTPUT_PATH, f"{imgname}.jpg")
        plt.savefig(output_image_path)

    def rotate(self) -> str:
        self.rotation_counter += 1
        return "RotateLeft"

    def explore(self, observations, attention, land_attention, room_attention) -> str:
        if not len(self.action_queue):
            self.action_queue = self.fbe.actions_toward_next_frontier()

        if len(self.action_queue) == 0:
            # Agent confused, move to spin mode
            self.fbe.reset()
            self.agent_mode = AgentMode.SPIN
            self.fbe.update_map(
                observations,
                self.glee_module.glee_caption[observations['object_goal']],
                attention,
                land_attention,  # added
                room_attention,  # added
                self.last_action,
                self.agent_mode)  # added)

            return self.rotate()

        return self.action_queue.pop(0)

    def exploit(self, observations, attention, land_attention, room_attention) -> str:

        if not len(self.action_queue):
            self.action_queue = self.fbe.action_towards_next_roi()

        if len(self.action_queue) == 0:
            # Agent confused, move to spin mode
            self.fbe.reset()
            self.agent_mode = AgentMode.SPIN
            self.fbe.update_map(
                observations,
                self.glee_module.glee_caption[observations['object_goal']],
                attention,
                land_attention,  # added
                room_attention,  # added
                self.last_action,
                self.agent_mode)  # added)

            return self.rotate()

        return self.action_queue.pop(0)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
from copy import copy, deepcopy
from enum import IntEnum
from heapq import heappop, heappush
from math import ceil, floor
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as nnf
import trimesh
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import AgglomerativeClustering
from torch import device

from src.shared.utils import tensor_linspace
from src.simulation.constants import (ACTION_SEQUENCES, ROTATION_MATRICIES,
                                      VOXEL_SIZE_M)
from src.simulation.utils import depth_frame_to_camera_space_xyz_thor_grid

#Context Map Generation Import

from src.models.prompting.ContextMapGeneration import CMGModule

#  PSL Python Import
from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
import pandas
from src.models.agent_mode import AgentMode

cmap = plt.get_cmap('jet')

try:
    import wandb
except ModuleNotFoundError:
    print("wandb not properly installed")

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class PQNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    # compares the second value
    def __lt__(self, other):
        return self.key < other.key

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))


class VoxelType(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    WALL = 3
    FRONTIER = 4
    # ROI = 5
    DBG = 5

    def color(self, rgba=False):

        c = None

        if self.value == VoxelType.UNKNOWN:
            c = [0, 225, 225]
        elif self.value == VoxelType.FREE:
            c = [255, 255, 255]
        elif self.value == VoxelType.OCCUPIED:
            c = [255, 0, 0]
        elif self.value == VoxelType.WALL:
            c = [255, 0, 0]
        elif self.value == VoxelType.FRONTIER:
            # c = [225, 0, 225]
            c = [255, 255, 255]
        # elif self.value == VoxelType.ROI:
        #     c = [255, 0, 255]
        elif self.value == VoxelType.DBG:
            c = [0, 255, 0]
        else:
            raise ValueError("Not supported enum")

        if rgba:
            c.append(225)

        return c


class FrontierPSLBasedExploration(object):

    def __init__(
            self,
            fov: float,
            device: device,
            max_ceiling_height: float,
            agent_height: float,
            agent_height_tolerance: float,
            rotation_deg: float,
            forward_dist: float,
            voxel_size_m: float,
            in_cspace: bool,
            wandb_log: bool,
            negate_action: bool,
            fail_stop: bool):

        super(FrontierPSLBasedExploration, self).__init__()
        self.fov = fov
        self.device = device
        self.camera_to_agent = torch.eye(4)
        self.max_ceiling_height = max_ceiling_height
        self.agent_height = agent_height
        self.agent_height_tolerance = agent_height_tolerance
        self.rotation_deg = rotation_deg
        self.forward_dist = forward_dist
        self.voxel_size_m = voxel_size_m
        self.in_cspace = in_cspace
        self.voxels = nx.Graph()
        self.floor_height_vox = int(
            math.floor(-self.agent_height / self.voxel_size_m))

        self.agent_voxel = None
        self.lookat_direction = torch.tensor([0., 0., 1.])
        self.exploration_targets = []
        self.visited_frontiers = set()

        self.run = None
        if wandb_log:
            self.run = wandb.init()

        self.reverse = True
        self.roi_targets = []
        self.failed_action = None
        self.last_observation = None
        self.negate_action = negate_action
        self.fail_stop = fail_stop
        self.PSL_infer = 'optim' #'optim' # one_hot


        self.CMG_module = CMGModule()
        self.prob_array_room = np.zeros(9)
        self.prob_array_obj = np.zeros(30)

        if self.PSL_infer == 'optim':
            self.psl_model = PSLModel('objnav3')
            # Add Predicates
            self.add_predicates(self.psl_model)

            # Add Rules
            self.add_rules(self.psl_model)


    def reset(self):
        self.camera_to_agent = torch.eye(4)
        self.voxels = nx.Graph()
        self.reverse = not self.reverse
        self.visited_frontiers = set()
        self.exploration_targets = []
        self.roi_targets = []
        self.failed_action = None
        self.last_observation = None

    def poll_roi_exists(self):
        return len(self.roi_targets) != 0

    def voxel_to_pointcloud(self, rgba=False, viz_edges=False, viz_roi_count=False, height_aware=False):
        voxel_groups = {}

        for v in self.voxels:
            voxel_type = self.voxels.nodes[v]["voxel_type"]

            if self.voxels.nodes[v]["obj_conf"] is None:
                obj_conf = 0.
            else:
                obj_conf = self.voxels.nodes[v]["obj_conf"]

            if voxel_type in voxel_groups:
                voxel_groups[voxel_type].append(
                    (v, obj_conf, self.voxels.nodes[v]["roi_count"]))
            else:
                voxel_groups[voxel_type] = [
                    (v, obj_conf, self.voxels.nodes[v]["roi_count"])]
        if viz_edges:
            for e in self.voxels.edges:
                if VoxelType.DBG in voxel_groups:
                    voxel_groups[VoxelType.DBG].append(
                        ((np.array(e[0]) + np.array(e[1])) / 2, 0, 0))
                else:
                    voxel_groups[VoxelType.DBG] = [
                        ((np.array(e[0]) + np.array(e[1])) / 2, 0, 0)]

        xyz = []
        color = []

        for t in voxel_groups:
            if t == VoxelType.OCCUPIED:
                for p, c, roi_count in voxel_groups[t]:
                    if height_aware:
                        h = self.voxels.nodes[p]['height']
                        if h is None:
                            h = self.floor_height_vox
                        xyz.append((p[0], h, p[2]))
                    else:
                        xyz.append(p)

                    if viz_roi_count:
                        if roi_count > 0:
                            color.append([int(255 * x) for x in cmap(1.0)])
                        else:
                            color.append([int(255 * x) for x in cmap(0.0)])
                    else:
                        color.append([int(255 * x) for x in cmap(c)])
            else:
                for p, _, _ in voxel_groups[t]:
                    xyz.append(p)
                    color.append(t.color(rgba=rgba))

        return xyz, color

    def dump_voxel_pointcloud(self, out_path):
        vertices, colors = self.voxel_to_pointcloud(rgba=True)

        ply = trimesh.points.PointCloud(
            vertices=np.array(vertices), colors=np.array(colors))

        # NOTE: uncomment for fig
        # with open('map.json', 'w') as f:
        #     json.dump(
        #         {
        #             'points': vertices,
        #             'colors': colors
        #         },
        #         f
        #     )
        ply.export(out_path)

    def log_voxel_pointcloud(self, suffix):
        if self.run is not None:
            vertices, colors = self.voxel_to_pointcloud(rgba=False)
            points_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]]
                                  for p, c in zip(vertices, colors)])
            self.run.log({f"point_cloud{suffix}": wandb.Object3D(points_rgb)})

    def update_map(self, observations, goal, attention, land_attention, room_attention, last_action, agent_mode):
        """Updates the map for frontier based exploration.

        Args:
            observations ([type]): [description]
            roi_mask ([type]): [description]
        """
        new_obs = torch.as_tensor(observations["depth"])
        self.goal = goal  # observations['object_goal']
        new_obs.squeeze_()
        self.agent_mode = agent_mode
        if self.last_observation is not None and self.fail_stop:
            abs_diff = torch.abs(self.last_observation-new_obs)
            m_pix = torch.mean(abs_diff)
            s_pix = torch.std(abs_diff)
            if m_pix.item() < 0.09 and s_pix.item() < 0.09:
                self.failed_action = last_action
            else:
                self.failed_action = None

        self.last_observation = new_obs

        # hflip as depth_frame_to_camera_space_xyz assumes left handed coordinate system
        # 원본 깊이 영상 (672, 672)  “nearest” 인터폴레이션 방식으로 크기 조절 (224,224)
        depth_lowres = nnf.interpolate(torch.as_tensor(observations["depth"]).squeeze().unsqueeze(0).unsqueeze(0),
                                       size=(224, 224),
                                       mode='nearest').squeeze()


        # [3, 224*224]
        # 깊이 영상(224,224)을 3차원 포인트 클라우드로 변환 (3,244)
        points_in_camera = depth_frame_to_camera_space_xyz_thor_grid(
            depth_frame=depth_lowres, mask=None, fov=self.fov
        )

        # 셀 깊이 값이 0.051보다 크면 빈틈이 없다고 판단해서 해당 영역을 True로 마스크
        no_holes = depth_lowres > 0.051

        # [1, 224*224]
        # Owl-vit 모델로부터 추출한 목표 물체 관심 영역 마스크 맵을 “nearest” 보간법으로 크기 조절 (224,224)
        clip_confidence = nnf.interpolate(attention.unsqueeze(0).unsqueeze(0),
                                          size=depth_lowres.shape,
                                          mode='nearest')[0]
        # added
        land_confidence = nnf.interpolate(land_attention.unsqueeze(0),
                                          size=depth_lowres.shape,
                                          mode='nearest')[0]
        room_confidence = nnf.interpolate(room_attention.unsqueeze(0),
                                          size=depth_lowres.shape,
                                          mode='nearest')[0]

        #3차원 포인트 클라우드에서 y축(수직)만 고려하여 에이전트가 인식할 수 있는 천장의 실제 높이 한계점을 설정
        # 각 포인트가 천장이 아니면 True, 천장이 맞으면 False임.
        not_ceiling_mask = points_in_camera[1, :, :] < (
            self.max_ceiling_height-self.agent_height-self.agent_height_tolerance)

        # “and” 연산을 통해 빈틈이 없고, 동시에 천장이 아닌 부분을 True로 마스킹 처리함
        # 해당 마스크 정보를 이용하여 3차원 포인트 클라우드와 목표 물체 관심 영역 마스크맵을 필터링함
        composite_mask = torch.logical_and(no_holes, not_ceiling_mask)

        points_in_camera = points_in_camera[:, composite_mask]

        # [224*224, 1]
        clip_confidence = clip_confidence[:, composite_mask].transpose(0, 1)
        # added
        land_confidence = land_confidence[:, composite_mask].transpose(0, 1)
        room_confidence = room_confidence[:, composite_mask].transpose(0, 1)

        # 에이전트 이전 행동에 따라 3차원 공간에서의 위치/회전 변환을 나타내는 4x4 변환 행렬임
        # 첫 번째 행  회전 후 x축이 어떻게 변화하였는지 표현 , 두 번째 행  y축 (그대로 유지) ,
        # 세 번째 행  z축의 변화를 표현, 네 번째 행  동차 좌표에서 변환 (이동이 없음을 나타냄)
        camera_new_to_old = torch.eye(4)

        if self.failed_action is None:
            camera_new_to_old = self._action_to_movement_matrix(last_action)

        self.camera_to_agent = self.camera_to_agent @ camera_new_to_old

        # 현재 에이전트의 회전 정보를 이용하여 바라보는 방향을 표현
        self.lookat_direction = self.camera_to_agent[:3,
                                                     :3] @ torch.tensor([0., 0., 1.])

        points_in_camera = torch.cat(
            (points_in_camera, torch.ones(1, points_in_camera.shape[1])), 0)

        # 에이전트 위치 복셀로 변환 과정
        agent_position = self.camera_to_agent[:3, 3]
        agent_voxel = agent_position / self.voxel_size_m
        agent_voxel[1] = self.floor_height_vox
        self.agent_voxel = tuple([v.item() for v in agent_voxel.int()])

        padding_in_camera = self._get_floor_padding_points(
            points_in_camera[:, points_in_camera[1, :] < 0]) #바닥 보간된 포인트 클라우드

        # 에이전트의 높이를 중심으로 바닥에 근접한 포인트들만 True로 표시
        floor_mask = torch.logical_and(points_in_camera[1, :] > (-self.agent_height-self.agent_height_tolerance),
                                       points_in_camera[1, :] < (-self.agent_height+self.agent_height_tolerance))

        # *** 카메라 좌표계에서 표현된 포인트들을 에이전트 좌표계로 변환하는 과정임 ***
        #  4x4 변환행렬은 카메라 좌표계에서의 3차원 포인트들 ***
        points_in_agent = self.camera_to_agent @ points_in_camera
        # *******************************************************

        local_voxels = self._voxelize_points(points_in_agent)
        heights = torch.clone(local_voxels[:, 1])

        # NOTE: uncomment for fig
        local_voxels[:, 1] = self.floor_height_vox

        # *** 기존 포인트 + 바닥까지 보간된 포인트들을 에이전트 좌표계로 변환하는 과정임 ***
        padding_in_agent = self.camera_to_agent @ padding_in_camera
        padding_voxels = self._voxelize_points(padding_in_agent)
        padding_voxels[:, 1] = self.floor_height_vox
        # ******************************************************************

        # NOTE: order of calls important here as state variables modified
        self._reset_dbg_to_free()
        self._reset_frontier()

        # *** local_voxels  VS  padding_voxels
        # local_voxels :
        # [1] 바닥이나 천장 같은 구조물뿐만 아니라, 다른 장애물이나 물체의 위치 정보도 포함함.
        # [2] 실제 에이전트 깊이 영상으로부터 추출함
        # --> 장애물 회피에 적합
        #
        # padding_voxels :
        # [1] 바닥 영역을 보다 세밀하게 채우기 위해 보간된 포인트 클라우드를 복셀화 함.
        # [2] 실제 에이전트의 깊이 영상으로부터 추출한 것이 아니라 바닥을 촘촘히 만들기 위해서 가상으로 생성된거임
        # --> 바닥 지형을 잘 파악할 수 있음

        self._fill_free(local_voxels[floor_mask])
        self._fill_free(padding_voxels, only_non_empty=True)

        self._fill_occupied(local_voxels, clip_confidence, land_confidence, room_confidence, heights, floor_mask)
        self._fill_frontier()
        # self._update_exploration_targets()
        self._update_roi_targets()

        # after all updating make sure that the agent is in a free space
        if self.agent_voxel not in self.voxels.nodes \
            or self.voxels.nodes[self.agent_voxel]["voxel_type"] != VoxelType.FREE:
            # 갱신 후, 에이전트의 현재 위치 셀 노드에 타입을 자유 영역으로 지정함
            self._fill_free(torch.tensor(self.agent_voxel).int().unsqueeze(0))

    def actions_toward_next_frontier(self):
        high_level_path = None

        if not len(self.exploration_targets):
            self._update_exploration_targets()

        for target in self.exploration_targets:
            if target in self.visited_frontiers:
                continue
            try:
                high_level_path = nx.astar_path(
                    self.voxels, self.agent_voxel, target, heuristic=self._node_dist, weight="weight")
                break
            except:
                continue

        if high_level_path is None:
            return []

        for n in high_level_path:
            self.voxels.nodes[n]['voxel_type'] = VoxelType.DBG

        curr = self.camera_to_agent[:3, 3]
        curr_lookat = self.lookat_direction

        target = None
        for waypoint in high_level_path:
            tmp = torch.tensor(
                waypoint, dtype=torch.float32) * self.voxel_size_m
            if self._node_dist(curr, tmp) > self.voxel_size_m * 3.5:
                target = tmp
                break
        if target is None:
            self.visited_frontiers.update(
                self._get_neighbors_twentyfour(high_level_path[-1]))
            self.visited_frontiers.add(high_level_path[-1])
            self.exploration_targets.pop(0)
            ########################################### -- Update Exploration Target
            self._update_exploration_targets()
            return self.actions_toward_next_frontier()

        curr[1] = target[1]

        local_expansion = self.low_level_planner(
            curr, curr_lookat, target)

        if local_expansion is None:
            return []

        return local_expansion.value["seq"][1:]

    def action_towards_next_roi(self):
        high_level_path = None
        for true_target, target in self.roi_targets:
            # assert target is not None
            if target is None:
                return ['Stop', ]

            if self._node_dist(self.agent_voxel, true_target) < 1/VOXEL_SIZE_M: #8
                bearing = np.array(true_target)
                bearing[1] = self.voxels.nodes[true_target]['height']

                agent_point = np.array(self.agent_voxel)
                agent_point[1] = 0

                correct_direction = bearing - agent_point
                correct_direction = correct_direction / np.sqrt(np.sum(correct_direction**2))
                lookat_direction = self.lookat_direction.numpy()
                y_rotation_options = [rot_mat @ lookat_direction for rot_mat in ROTATION_MATRICIES]
                cosines = [np.dot(option, correct_direction) for option in y_rotation_options]
                index1 = np.argmax(cosines)

                seq = list(ACTION_SEQUENCES[index1][0][:-1])

                return seq + ['Stop', ]

        # for true_target, target in self.roi_targets:
            try:
                high_level_path = nx.astar_path(
                    self.voxels, self.agent_voxel, target, heuristic=self._node_dist, weight="weight")
                break
            except:
                continue

        if high_level_path is None:
            for true_target, target in self.roi_targets:
                try:
                    exploration_targets = copy(self.exploration_targets)
                    exploration_targets = sorted(
                        exploration_targets,
                        key=lambda x: self._node_dist(true_target, x),
                    )

                    found_path = False
                    for exploration_target in exploration_targets:
                        try:
                            high_level_path = nx.astar_path(
                                self.voxels, self.agent_voxel, exploration_target, heuristic=self._node_dist, weight="weight")
                            found_path = True
                            break
                        except:
                            continue

                    if found_path:
                        break
                except Exception as e:
                    continue

        # have tried to nav to target directly and to a frontier, both unsuccessfully so give up
        if high_level_path is None:
            return []

        for n in high_level_path:
            self.voxels.nodes[n]['voxel_type'] = VoxelType.DBG

        curr = self.camera_to_agent[:3, 3]
        curr_lookat = self.lookat_direction

        target = None
        for waypoint in high_level_path:
            tmp = torch.tensor(
                waypoint, dtype=torch.float32) * self.voxel_size_m
            if self._node_dist(curr, tmp) > self.voxel_size_m * 3.5:
                target = tmp
                break

        if target is not None:
            curr[1] = target[1]

        if target is None:
            # NOTE: case where there is not path to the target
            return ['RotateLeft', ]

        local_expansion = self.low_level_planner(
            curr, curr_lookat, target)

        if local_expansion is None:
            return []

        return local_expansion.value["seq"][1:]

    def low_level_planner(self, curr, curr_lookat, target, max_tree_depth=50):
        pq = []

        heappush(pq, PQNode(self._node_dist(curr, target).item(),
                 {"seq": ["Start"], "seq_position": [torch.clone(curr)], "position": torch.clone(curr), "lookat": torch.clone(curr_lookat)}))
        visited = set()
        for i in range(max_tree_depth):
            node = None
            try:
                node = deepcopy(heappop(pq))
            except:
                return None

            # expand the neighbors and if valid queue them up
            for seq, (rot_deg, trans_m) in ACTION_SEQUENCES:
                if self.failed_action is not None:
                    if seq[0] == self.failed_action:
                        continue

                value = deepcopy(node.value)
                value["seq"] = value["seq"] + list(seq)

                rot_mat = torch.tensor(R.from_euler(
                    "y", rot_deg, degrees=True).as_matrix(), dtype=torch.float32)

                value["lookat"] = rot_mat @ value["lookat"]
                value["position"] += (value["lookat"] * trans_m)

                subvox = tuple(
                    (value["position"] / self.voxel_size_m).tolist())
                vox = (round(subvox[0]), round(subvox[1]), round(subvox[2]))
                if vox in visited:
                    continue
                visited.add(vox)

                surrounding_ops = (
                    # (ceil, int, floor),
                    # (ceil, int, ceil),
                    # (floor, int, floor),
                    # (floor, int, ceil),
                    (round, round, round),
                )

                valid_position = True
                for op in surrounding_ops:
                    candidate = (op[0](subvox[0]), op[1](
                        subvox[1]), op[2](subvox[2]))

                    if candidate in self.voxels and self.voxels.nodes[candidate]["voxel_type"] in [VoxelType.OCCUPIED, VoxelType.WALL]:
                        valid_position = False

                if not valid_position:
                    continue

                key = self._node_dist(value["position"], target).item()

                if key <= (self.voxel_size_m * 3.5):
                    self.failed_action = None
                    return PQNode(key, value)

                heappush(pq, PQNode(key, value))

            # reset failed action past level 1 depth.
            self.failed_action = None

        return None

    def _action_to_movement_matrix(self, action):
        action_delta = torch.eye(4)

        negation_factor = 1
        if self.negate_action:
            negation_factor = -1

        if action == "RotateLeft":
            action_delta[:3, :3] = torch.tensor(
                R.from_euler("y", negation_factor * self.rotation_deg, degrees=True).as_matrix())
        elif action == "RotateRight":
            action_delta[:3, :3] = torch.tensor(
                R.from_euler("y", negation_factor * -self.rotation_deg, degrees=True).as_matrix())
        elif action == "MoveAhead":
            action_delta[2, 3] = self.forward_dist
        elif action is None:
            pass
        elif action == "TeleportFull":
            pass
        elif action == "Teleport":
            pass
        else:
            raise ValueError("unsupported action type")

        return action_delta

    def add_rules(self, model):
        # if self.args.reasoning in ['both', 'obj']:
        model.add_rule(Rule('1: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('1: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        # # if self.args.reasoning in ['both', 'room']:
        model.add_rule(Rule('1: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('1: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('1: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))

    def add_predicates(self, model):

        predicate = Predicate('IsNearObj', closed=True, size=2)
        model.add_predicate(predicate)
        #
        predicate = Predicate('ObjCooccur', closed=True, size=1)
        model.add_predicate(predicate)

        predicate = Predicate('IsNearRoom', closed=True, size=2)
        model.add_predicate(predicate)

        predicate = Predicate('RoomCooccur', closed=True, size=1)
        model.add_predicate(predicate)

        predicate = Predicate('Choose', closed=False, size=1)
        model.add_predicate(predicate)

        predicate = Predicate('ShortDist', closed=True, size=1)
        model.add_predicate(predicate)

    def create_voxels_bbox(self,coordinates):
        if coordinates:
            x_coords, y_coords = zip(*[[coord[0][0], coord[0][1]] for coord in coordinates])
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            return (min_x, min_y), (max_x, max_y)
        return None

    def _update_exploration_targets(self):

        frontiers = self._cluster_frontiers()

        frontier_means = {k: np.mean(frontiers[k], axis=0) for k in frontiers}

        frontier_diffs = {k: [self._node_dist(
            p, frontier_means[k]) for p in frontiers[k]] for k in frontiers}

        exploration_targets = {k: frontiers[k][np.argmin(
            frontier_diffs[k])] for k in frontiers}

        selected_frontier_list = [sublist for sublist in exploration_targets.values()]
        num_frontiers = len(selected_frontier_list)
        scores = np.zeros((num_frontiers))

        frontier_agent_distance = np.array([self._node_dist(f, self.agent_voxel) for f in selected_frontier_list])

        frontier_agent_distance_inverse = 1 - (np.clip(frontier_agent_distance, 1.2, 11.2) - 1.2) / (11.2 - 1.2)

        # if self.agent_mode == AgentMode.EXPLORE:

        room_node_hash = {}
        object_node_hash = {}
        for v in self.voxels:
            room_conf = self.voxels.nodes[v]['room_conf']
            land_conf = self.voxels.nodes[v]['land_conf']
            if room_conf.any():
                max_conf_idx = torch.argmax(room_conf).item()
                max_conf_value = room_conf[max_conf_idx]

                if max_conf_idx not in room_node_hash:
                    room_node_hash[max_conf_idx] = []
                room_node_hash[max_conf_idx].append([[v[0], v[2], max_conf_value]])

            if land_conf.any():
                max_conf_idx = torch.argmax(land_conf).item()
                max_conf_value = land_conf[max_conf_idx]

                if max_conf_idx not in object_node_hash:
                    object_node_hash[max_conf_idx] = []
                object_node_hash[max_conf_idx].append([[v[0], v[2], max_conf_value]])

        # print(room_node_hash)

        self.CMG_module.room_map_update_v2(room_node_hash)
        self.CMG_module.obj_map_update_v2(object_node_hash)
        self.CMG_module.agent_voxel_update(self.agent_voxel)
        (llm_prob_array_obj, llm_prob_array_room,
         analysis_room_voxels, analysis_object_voxels, experiments_config) = self.CMG_module.start_process_v2(self.goal)

        if not np.all(llm_prob_array_room == 0):
            self.prob_array_room = llm_prob_array_room

        if not np.all(llm_prob_array_obj == 0):
            self.prob_array_obj = llm_prob_array_obj

        if self.PSL_infer == 'optim':
            for predicate in self.psl_model.get_predicates().values():
                if predicate.name() in ['ROOMCOOCCUR']:
                    predicate.clear_data()

            data = pandas.DataFrame([[i, self.prob_array_room [i]] for i in range(len(self.prob_array_room))],
                                    columns=list(range(2)))
            self.psl_model.get_predicate('RoomCooccur').add_data(Partition.OBSERVATIONS, data)

            for predicate in self.psl_model.get_predicates().values():
                if predicate.name() in ['OBJCOOCCUR']:
                    predicate.clear_data()

            data = pandas.DataFrame([[i, self.prob_array_obj[i]] for i in range(len(self.prob_array_obj))],
                                    columns=list(range(2)))
            self.psl_model.get_predicate('ObjCooccur').add_data(Partition.OBSERVATIONS, data)

        frontier_near_room = {}
        frontier_near_obj = {}
        nf_thres = 6 # Room - Frontier Near Threshold

        for room_index, coordinates in room_node_hash.items():
            room_voxel_bbox = self.create_voxels_bbox(coordinates)

            for target in selected_frontier_list:

                whether_near_room = np.zeros(9)
                if room_voxel_bbox and experiments_config['context_map']['configure'] != 'grid':
                    target_x, target_y = target[0], target[2]
                    (min_x, min_y), (max_x, max_y) = room_voxel_bbox
                    if min_x - nf_thres <= target_x <= max_x + nf_thres and min_y - nf_thres <= target_y <= max_y + nf_thres:
                        whether_near_room[room_index] = coordinates[0][0][-1].item()

                frontier_near_room[target] = whether_near_room

        for obj_index, object_voxels in object_node_hash.items():
            whether_near_obj = np.zeros(len(selected_frontier_list))

            for tar_idx, target in enumerate(selected_frontier_list):

                for obj_voxel in object_voxels:

                    obj_frontier_dist = self._node_dist(target, (obj_voxel[0][0], 0, obj_voxel[0][1]))
                    # print(obj_frontier_dist)

                    if obj_frontier_dist < 10 and experiments_config['context_map']['configure'] != 'grid':
                        whether_near_obj[tar_idx] = obj_voxel[0][-1]
            # print(whether_near_obj)
            frontier_near_obj[obj_index] = whether_near_obj

        try:
            if self.PSL_infer == 'optim':
                for predicate in self.psl_model.get_predicates().values():
                    if predicate.name() in ['ISNEARROOM']:
                        predicate.clear_data()

            for i, loc in enumerate(selected_frontier_list):
                whether_near_room_list = frontier_near_room[loc].tolist()

                if self.PSL_infer == 'optim':
                    data = pandas.DataFrame(
                        [[j, i, whether_near_room_list[j]] for j in range(len(whether_near_room_list))],
                        columns=list(range(3)))

                    self.psl_model.get_predicate('IsNearRoom').add_data(Partition.OBSERVATIONS, data)
                else:
                    score_1 = np.clip(1 - (1 - self.prob_array_room) - (1 - np.array(whether_near_room_list)), 0, 10)
                    score_2 = 1 - np.clip(self.prob_array_room + (1 - np.array(whether_near_room_list)), -10, 1)
                    scores[i] = np.sum(score_1) - np.sum(score_2)

            if self.PSL_infer == 'optim':
                for predicate in self.psl_model.get_predicates().values():
                    if predicate.name() in ['ISNEAROBJ', 'CHOOSE', 'SHORTDIST']:
                        predicate.clear_data()

            for i in range(30):

                if i not in frontier_near_obj:
                    whether_near_obj = np.zeros(len(selected_frontier_list))
                    whether_near_obj_list = whether_near_obj.tolist()
                else:
                    whether_near_obj_list = frontier_near_obj[i].tolist()

                if self.PSL_infer == 'optim':
                    data = pandas.DataFrame(
                        [[i, j, whether_near_obj_list[j]] for j in range(len(whether_near_obj_list)) if
                         whether_near_obj_list[j] > 0], columns=list(range(3)))
                    # print(data)
                    self.psl_model.get_predicate('IsNearObj').add_data(Partition.OBSERVATIONS, data)

                else:
                    score_1 = np.clip(1 - (1 - self.prob_array_obj[i]) - (1 - np.array(whether_near_obj_list)), 0, 10)
                    score_2 = 1 - np.clip(self.prob_array_obj[i] + (1 - np.array(whether_near_obj_list)), -10, 1)
                    scores += score_1 - score_2

            if self.PSL_infer == 'optim':
                data = pandas.DataFrame([[i] for i in range(num_frontiers)], columns=list(range(1)))
                self.psl_model.get_predicate('Choose').add_data(Partition.TARGETS, data)

                data = pandas.DataFrame([[i, frontier_agent_distance_inverse[i]] for i in range(num_frontiers)],
                                        columns=list(range(2)))
                self.psl_model.get_predicate('ShortDist').add_data(Partition.OBSERVATIONS, data)
                result = self.psl_model.infer(additional_cli_options=ADDITIONAL_CLI_OPTIONS,
                                              psl_config=ADDITIONAL_PSL_OPTIONS)
                for key, value in result.items():
                    result_dt_frame = value

                scores = result_dt_frame.loc[:, 'truth']
                frontier_sorted = scores.sort_values(ascending=False)
                self.exploration_targets = [selected_frontier_list[i] for i in frontier_sorted.index.tolist()]

            else:
                scores += 2 * frontier_agent_distance_inverse
                frontier_sorted = np.argsort(scores)[::-1]
                self.exploration_targets = [selected_frontier_list[i] for i in frontier_sorted.tolist()]
                # print(scores)

            self.visualize_voxels(analysis_room_voxels, analysis_object_voxels, selected_frontier_list,
                                  self.exploration_targets[0])

        except:
            print("UnboundLocalError: local variable 'result_dt_frame' referenced before assignment")
            self.exploration_targets = []

        # else:
        #
        #     self.exploration_targets = sorted(
        #         exploration_targets.values(),
        #         key=lambda x: self._node_dist(self.agent_voxel, x),
        #         reverse=self.reverse)

    def _update_roi_targets(self):
        for i in range(len(self.roi_targets)):
            roi, closest_free = self.roi_targets[i]
            none_check = closest_free is None

            existence_check = False
            if not none_check:
                existence_check = closest_free in self.voxels.nodes

            not_free_check = False
            if existence_check:
                not_free_check = self.voxels.nodes[closest_free]["voxel_type"] in [
                    VoxelType.OCCUPIED, VoxelType.WALL]

            if none_check or not_free_check:

                dist = self._node_dist(self.agent_voxel, roi)
                line = tensor_linspace(torch.tensor(roi).float(),
                                       torch.tensor(self.agent_voxel).float(),
                                       ceil(dist*2))
                for j in range(line.shape[1]):
                    candidate = tuple(torch.round(line[:, j]).int().tolist())
                    if candidate in self.voxels.nodes and self.voxels.nodes[candidate]["voxel_type"] not in [VoxelType.OCCUPIED, VoxelType.WALL]:
                        self.roi_targets[i][1] = candidate
                        break

    def _node_dist(self, a, b):
        (x1, _, z1) = a
        (x2, _, z2) = b
        dis = ((x1 - x2) ** 2 + (z1 - z2) ** 2) ** 0.5
        # print(dis)
        return ((x1 - x2) ** 2 + (z1 - z2) ** 2) ** 0.5

    def _voxelize_points(self, points):
        voxels = torch.round(points[:3, :] / self.voxel_size_m).int()
        voxels = torch.transpose(voxels, 0, 1)

        return voxels

    def _get_floor_padding_points(self, points_in_camera):
        # 포인트 복셀화 #
        voxels_in_camera = self._voxelize_points(points_in_camera)
        # 바닥 높이 설정 --> 모든 복셀들이 동일한 높이(바닥 위치)에 위치하도록 함! #
        voxels_in_camera[:, 1] = self.floor_height_vox
        #에이전트 위치를 나타내는 복셀 추가함
        voxels_in_camera = torch.cat(
            (voxels_in_camera, torch.tensor([list(self.agent_voxel)])), dim=0)
        #중복된 복셀 제거 --> 계산 효율성 높이기 위함
        voxels = torch.unique(voxels_in_camera, dim=0)

        # ***** 바닥과 가장 가까운 포인트들을 식별하는 과정 ***** #
        min_z = {}
        for i in range(voxels.shape[0]):
            if voxels[i][0].item() not in min_z:
                min_z[voxels[i][0].item()] = voxels[i].float()
            else:
                if voxels[i][2].item() < min_z[voxels[i][0].item()][2]:
                    min_z[voxels[i][0].item()] = voxels[i].float()
        # ********************************************** #

        # **** 바닥에 가까운 포인트들 사이에 0.25 간격만큼 포인트들을 보간하는 과정 **** #
        step = 0.25
        interpolated_points = None
        for k in min_z: #
            dist = torch.norm(min_z[k]) #모든 값(x,y,z)들을 모두 제곱해서 더한 후 제곱근을 씌움

            num_steps = torch.ceil(dist/step).int() #0.25 간격으로 몇 개의 보간 포인트를 생성할지 결정함

            padding = tensor_linspace(torch.tensor(
                [0., 0., 0.]), min_z[k], num_steps) #min_z[k]까지 num_steps 개수만큼 포인트를 균일하게 생성함 --> 중간 포인트들을 생성

            if interpolated_points is None:
                interpolated_points = padding
            else:
                interpolated_points = torch.cat(
                    (interpolated_points, padding), 1)
    # **** ***************************************** **** #

        interpolated_points *= self.voxel_size_m #모든 보간된 포인트를 실제 공간 크기(복셀 크기)에 맞게 조정함

        interpolated_points[1, :] = -self.agent_height #에이전트의 높이를 고려하여 보간 포인트를 바닥에 맞추는 과정임

        interpolated_points = torch.cat(
            (interpolated_points, torch.ones(1, interpolated_points.shape[1])), 0)

        return interpolated_points

    def _cluster_frontiers(self):
        """Runs agglomerative clustering on frontiers
        """
        dust = []
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FRONTIER:
                count = 0

                # 프론티어 영역 노드셀 기준 8방위에 존재하는 인접셀 추출
                neighbors, _ = self._get_neighbors_eight(v)


        # *** 인접셀들 중에 자유 영역이 하나도 없고, 에이전트가 위치한 복셀이 아니라면 해당 프론티어 영역 노드셀 제거함  ***
        # 자유 영역이 없으면 해당 프론티어로 가지를 못하니까 타겟으로 잡아봤자 의미가 없다라는 생각을 하는 듯
        # --> ESC와 차별점이 될 수도 있음. 고스트 타겟을 제거하는 방법 중 하나..?
                for n in neighbors:
                    if n in self.voxels and self.voxels.nodes[n]["voxel_type"] == VoxelType.FREE:
                        count += 1

                if count == 0 and v != self.agent_voxel:
                    dust.append(v)

        for d in dust:
            self.voxels.remove_node(d)
        # ******************************************************************

        # 이제 진짜 프론티어 후보를 결정
        candidates = []
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FRONTIER:
                candidates.append(v)

        frontiers = {}

        # 만약에, 프론티어 셀 후보가 2개 이하면, 해당 셀은 자유 셀이라고 간주하고, 프론티어 없음
        if len(candidates) < 3:
            for v in candidates:
                self.voxels.nodes[v]["voxel_type"] = VoxelType.FREE
            return frontiers

        # *** 각 프론티어 영역 셀 노드들을 하나의 클러스터로 지정하고, 1.5 (?)이내에 있는 인접 셀 노드들을 점진적으로 병합 ***
        # 병합은 1.5(?)를 초과하지 않는 한 계속 진행함
        # https://woolulu.tistory.com/48

        clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage="single",
            distance_threshold=1.5
        ).fit(candidates)

        for i, c in enumerate(clustering.labels_):
            if c in frontiers:
                frontiers[c].append(candidates[i])
            else:
                frontiers[c] = [candidates[i]]

        return frontiers

    def _fill_occupied(self, local_voxels, conf, l_conf, r_conf, heights, mask):
        # added

        if local_voxels is None:
            return

        local_vox_unique = None
        inverse_indices = None
        conf_masked = None
        l_conf_masked = None
        r_conf_masked = None
        heights_masked = None

        if mask is not None:
            # 바닥 높이에 해당하지 않는 복셀들을 필터링함.
            # 즉, 바닥이 아닌 다른 구조물이나 물체에 해당하는 복셀들을 대상!
            local_vox_unique, inverse_indices = torch.unique(local_voxels[~mask], dim=0, return_inverse=True)
            conf_masked = conf[~mask]
            l_conf_masked = l_conf[~mask]
            r_conf_masked = r_conf[~mask]
            heights_masked = heights[~mask]

        else:
            local_vox_unique, inverse_indices = torch.unique(local_voxels, dim=0, return_inverse=True)
            conf_masked = conf
            l_conf_masked = l_conf
            r_conf_masked = r_conf
            heights_masked = heights

        # NOTE: this throws for some reason in some cases: IndexError: too many indices for tensor of dimension 1
        #       debug after deadline

        groups = []
        local_voxels_unique = local_vox_unique.int()
        for i in range(local_voxels_unique.shape[0]):
            k1 = tuple(local_voxels_unique[i, :].tolist())  # ex. k1: (-6, -7. 6)

            inv_mask = inverse_indices == i

            conf_inv_mask = conf_masked[inv_mask]
            l_conf_inv_mask = l_conf_masked[inv_mask]
            r_conf_inv_mask = r_conf_masked[inv_mask]
            height_inv_mask = heights_masked[inv_mask]

            j = torch.argmax(conf_inv_mask)
            k2 = conf_inv_mask[j]

            # j = torch.argmax(height_inv_mask)
            k3 = height_inv_mask[j]

            j1 = torch.argmax(l_conf_inv_mask)
            k4 = l_conf_inv_mask[j1 // l_conf.shape[1], ]

            j2 = torch.argmax(r_conf_inv_mask)
            k5 = r_conf_inv_mask[j2 // r_conf.shape[1], ]

            groups.append((k1, k2, k3, k4, k5))

        observation_rois = []
        for v_tuple, v_conf, v_height, v_l_conf, v_r_conf in groups:
            # print(v_l_conf)
            if v_tuple in self.voxels.nodes:
                if self.voxels.nodes[v_tuple]["voxel_type"] == VoxelType.WALL:  # or\
                    # self.voxels.nodes[v_tuple]["voxel_type"] == VoxelType.ROI:
                    continue

            # 객체나 구조물 중 하나를 구성하고 있기 때문에,, 해당 복셀값은 OCCUPIED 라는 판단이 가능함!
            self._add_node_conditional(
                v_tuple, v_conf.item(), v_l_conf, v_r_conf, v_height.item(), VoxelType.OCCUPIED)

            # now check if we have an ROI so that we can switch to exploit mode
            # 점유된 영역이 관심 영역(ROI)이 한 개라도 있으면 Exploit 모드로 전환할까? 말까를 고민해야 되기 때문에,,
            # 우선 observation_rois 리스트에 넣고, 추후에 Exploit 할 지 말지 결정함!!
            if self.voxels.nodes[v_tuple]["roi_count"] > 0:
                observation_rois.append(v_tuple)

            # 해당 복셀은 점유된 상황이니까, 이동 불가능 하도록 사전에 연결된 인접 노드들 간의 간선 연결을 제거함
            for edge in list(self.voxels.edges(v_tuple)):
                self.voxels.remove_edge(*edge)

            if self.in_cspace:
                neighbors, _ = self._get_neighbors_four(v_tuple)
                for vc_tuple in neighbors:
                    if vc_tuple in self.voxels.nodes:
                        if self.voxels.nodes[vc_tuple]["voxel_type"] == VoxelType.WALL:  # or\
                            # self.voxels.nodes[vc_tuple]["voxel_type"] == VoxelType.ROI:
                            continue

                    # 해당 복셀은 점유된 상황이니까, 상,하,좌,우에 존재하는 인접 노드들을 OCCUPIED로 지정하고,
                    # 그들 간의 간선 연결을 제거함

                    # 점유된 복셀 주변을 아예 못가도록 더 확장시키는 방법인 듯??
                    # 점유된 복셀 왼쪽에 자유 영역인 셀이 있더라도,,, 그 셀은 점유되었다고 판단함
                    self._add_node_conditional(
                        vc_tuple, None, torch.zeros(30), torch.zeros(9), self.floor_height_vox, VoxelType.OCCUPIED)

                    for edge in list(self.voxels.edges(vc_tuple)):
                        self.voxels.remove_edge(*edge)

        if len(observation_rois):
            # sort in order of most seen roi, break ties with euclidean distance to the point.
            # grab first entry as the target
            observation_roi_target = sorted(observation_rois, key=lambda x: (
                self._node_dist(self.agent_voxel, x), -self.voxels.nodes[x]["roi_count"]), reverse=False)[0]
            exists = [k1 for k1, _ in self.roi_targets]
            if observation_roi_target not in exists:
                self.roi_targets.append([observation_roi_target, None])
                self.roi_targets = sorted(self.roi_targets, key=lambda x: self._node_dist(self.agent_voxel, x[0]), reverse=False)

    def _fill_free(self, local_voxels, only_non_empty=False):
        edges_to_add = {}

        # 여러 포인트들이 동일한 복셀 공간에 할당될 경우에 중복된 포인트들이 존재 가능함 -> torch.unique !
        local_voxels_unique = torch.unique(local_voxels, dim=0)
        for v in local_voxels_unique:
            v_tuple = tuple(v.tolist())
            if only_non_empty and v_tuple in self.voxels.nodes:
                continue
            # if v_tuple in self.voxels.nodes and self.voxels.nodes[v_tuple]["voxel_type"] == VoxelType.ROI:
            #     continue

            # 8 방위 (상,하,좌,우 + 대각선)에 대한 인접 생성
            # 인접 노드들은 해당 복셀이 연결될 수 있음을 나타냄
            # 상, 하, 좌, 우 이동은 1.0의 비용을, 대각선 이동은 1.41의 비용을 나타냄
            # --> 경로 계획 시 복셀 간 이동 비용을 계산하는데 사용함
            edges_to_add[v_tuple] = self._get_neighbors_eight(v_tuple)

            # 복셀이 자유 공간
            self._add_node_conditional(v_tuple, 0., torch.zeros(30), torch.zeros(9), self.floor_height_vox, VoxelType.FREE)

        # 에이전트와 인접 복셀들 간의 연결 관계 정의
        if self.agent_voxel not in self.voxels.nodes:
            edges_to_add[self.agent_voxel] = self._get_neighbors_eight(
                self.agent_voxel)
            # 자유 공간
            self._add_node_conditional(self.agent_voxel, 0., torch.zeros(30), torch.zeros(9), self.floor_height_vox, VoxelType.FREE)


        for src in edges_to_add:
            for i in range(len(edges_to_add[src][0])):
                sink = edges_to_add[src][0][i]
                edge_weight = edges_to_add[src][1][i]

                if sink in self.voxels and self.voxels.nodes[sink]["voxel_type"] == VoxelType.FREE:
                    # 인접 복셀이 자유 영역이라면 이동 가능하기 때문에 간선 연결함!
                    self.voxels.add_edge(src, sink, weight=edge_weight)

    def _reset_frontier(self):
        """Sets all frontier space to free space.
        """
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FRONTIER:
                self.voxels.nodes[v]["voxel_type"] = VoxelType.FREE

    def _reset_dbg_to_free(self):
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.DBG:
                self.voxels.nodes[v]["voxel_type"] = VoxelType.FREE

    def _fill_frontier(self):
        """Looks for the edge of free space and unknown to set fontier.
        """
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FREE:
                count = 0

                #자유영역 노드 셀의 상,하,좌,우에 존재하는 인접 노드셀 추출
                neighbors, _ = self._get_neighbors_four(v)

                #
                for n in neighbors:

                    # 인접 노드셀을 하나씩 순회하면서, Unknown 셀노드가 몇 개 있는지 체크함
                    # 자유 영역 노드셀 주변에 다 Known 노드셀이면 count 값은 4임. 이땐 프론티어가 아님
                    if n in self.voxels:
                        count += 1

                if count != 4:
                    # 자유 영역 노드셀 주변에 하나라도 Unknwon 셀이 존재하면, 해당 셀 노드를 프론티어 타입 지정
                    self.voxels.nodes[v]["voxel_type"] = VoxelType.FRONTIER

    def _get_neighbors_four(self, voxel_tuple: Tuple[float]) -> List[Tuple[float]]:
        """Gets four (plus) neighbors of a voxel in xz space.

        Args:
            voxel_tuple (Tuple[float]): center voxel.

        Returns:
            List[Tuple[float]]: neighbors
        """
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        edge_costs = (1., 1., 1., 1.)
        nodes = []
        for o in offsets:
            nodes.append(
                (voxel_tuple[0] + o[0], voxel_tuple[1], voxel_tuple[2] + o[1]))

        return nodes, edge_costs

    def _get_neighbors_eight(self, voxel_tuple: Tuple[float]) -> List[Tuple[float]]:
        """Gets eight (cross and plus) neighbors of a voxel in xz space.

        Args:
            voxel_tuple (Tuple[float]): center voxel.

        Returns:
            List[Tuple[float]]: neighbors
        """
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, 1), (-1, -1), (1, -1))
        edge_costs = (1., 1., 1., 1., 1.41, 1.41, 1.41, 1.41)

        nodes = []
        for o in offsets:
            nodes.append(
                (voxel_tuple[0] + o[0], voxel_tuple[1], voxel_tuple[2] + o[1]))

        return nodes, edge_costs

    def _get_neighbors_twentyfour(self, voxel_tuple: Tuple[float]) -> List[Tuple[float]]:
        """Gets eight (cross and plus) neighbors of a voxel in xz space.

        Args:
            voxel_tuple (Tuple[float]): center voxel.

        Returns:
            List[Tuple[float]]: neighbors
        """
        nodes = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                nodes.append(
                    (voxel_tuple[0] + i, voxel_tuple[1], voxel_tuple[2] + j))

        return nodes

    def _add_node_conditional(self, voxel_tuple, obj_conf, land_conf, room_conf, height, type):
        # added

        is_roi = False
        if obj_conf is not None:
            is_roi = obj_conf > 0.9
            if is_roi:
                print("GOAL FOUND!!!!!!!!!!!!!")

        # if land_conf.any() > 0:
        #      tmp = land_conf

        if voxel_tuple not in self.voxels:
            self.voxels.add_node(voxel_tuple,
                                 voxel_type=type,
                                 obj_conf=obj_conf,
                                 land_conf=land_conf,    # is this right?
                                 room_conf=room_conf,    # is this right?
                                 roi_count=int(is_roi),
                                 height=height)
        else:
            self.voxels.nodes[voxel_tuple]["voxel_type"] = type
            if self.voxels.nodes[voxel_tuple]["obj_conf"] is None:
                self.voxels.nodes[voxel_tuple]["obj_conf"] = obj_conf
                self.voxels.nodes[voxel_tuple]["height"] = height
            else:
                if obj_conf is not None:
                    if obj_conf >= self.voxels.nodes[voxel_tuple]["obj_conf"]:
                        self.voxels.nodes[voxel_tuple]["obj_conf"] = obj_conf
                        self.voxels.nodes[voxel_tuple]["height"] = height

            if self.voxels.nodes[voxel_tuple]["land_conf"] is None:
                if land_conf is not None:
                    # print(land_conf)
                    for i in range(len(land_conf.tolist())):
                        self.voxels.nodes[voxel_tuple]["land_conf"][i] = land_conf[i].item()
            else:
                if land_conf is not None:
                    for i in range(len(land_conf.tolist())):
                        if land_conf[i].item() >= self.voxels.nodes[voxel_tuple]["land_conf"][i]:
                            self.voxels.nodes[voxel_tuple]["land_conf"][i] = land_conf[i].item()  # fill with max score

            if self.voxels.nodes[voxel_tuple]["room_conf"] is None:
                if room_conf is not None:
                    for i in range(len(room_conf.tolist())):
                        self.voxels.nodes[voxel_tuple]["room_conf"][i] = room_conf[i].item()
            else:
                if room_conf is not None:
                    for i in range(len(room_conf.tolist())):
                        if room_conf[i].item() >= self.voxels.nodes[voxel_tuple]["room_conf"][i]:
                            self.voxels.nodes[voxel_tuple]["room_conf"][i] = room_conf[i].item()  # fill with max score

            self.voxels.nodes[voxel_tuple]["roi_count"] += int(is_roi)

        # tmp = self.voxels.nodes[voxel_tuple]
        # print(self.voxels.nodes[voxel_tuple]["land_conf"].any()>0)


    def visualize_voxels(self, room_voxels, object_voxels, selected_frontiers, top_frontier):
        fig, ax = plt.subplots(figsize=(10, 8))

        # 방 복셀 시각화
        room_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(room_voxels)))
        for idx, (room_name, coords) in enumerate(room_voxels.items()):
            ax.scatter(coords[:, 0], coords[:, 1], marker='s', color=room_colors[idx], label=room_name, alpha=0.6)

        #
        object_colors = plt.cm.Accent(np.linspace(0, 1, len(object_voxels)))
        for idx, (object_name, coords) in enumerate(object_voxels.items()):
            ax.scatter(coords[:, 0], coords[:, 1], marker='p', s=110, color=object_colors[idx], label=object_name, alpha=0.9)

        # 프론티어 위치
        frontier_coords = np.array(selected_frontiers)[:, [0, 2]]
        ax.scatter(frontier_coords[:, 0], frontier_coords[:, 1], marker='^', s=100, c='red', label='Frontiers',
                   alpha=0.8)

        # 탑 프론티어
        ax.scatter(top_frontier[0],top_frontier[2], marker='*', s=110, c='blue', label='Top Frontier', alpha=0.8)

        # 에이전트 위치
        ax.scatter(self.agent_voxel[0], self.agent_voxel[2], marker='o', s=100, c='green', label='Agent Position',
                   alpha=0.8)

        ax.set_xlabel('X ')
        ax.set_ylabel('Z ')

        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)

        ax.legend()
        ax.grid(True)

        plt.show()

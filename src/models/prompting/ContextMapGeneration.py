
import numpy as np
from collections import defaultdict
import sys
import copy
import math

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean

from src.models.prompting.LLM_Prompting import CLPModule

from torch import tensor
from src.simulation.constants import THOR_LANDMARK_TYPES, THOR_ROOM_TYPES
import yaml

class CMGModule:

    def __init__(self):
        # self.load_npy('/home/baebro/robothor_cp_240312/robothor_cp/robothor_cp/temp_file')
        # self.temp_init()
        self.error_analysis = True
        self.visualization = True
        self.map_size = 600
        self.room_select_threshold = 0.6
        self.ins_room_cout = -1
        self.node_idx = 0
        self.experiments_config = self.load_config('experiments_config.yaml')
        self.relation_labels = ['in', 'left of', 'right of', 'behind', 'front', 'near', 'next to', 'positioned at']

        self.spatial_relations = ['near', 'far', 'left', 'right', 'northeast', 'northwest', 'southeast', 'southwest']

        self.categories_rooms = THOR_ROOM_TYPES
        self.categories_objects = THOR_LANDMARK_TYPES

        use_llm = "Llama2"
        self.llm = CLPModule(self.categories_objects, self.categories_rooms, use_llm)

    def load_npy(self,file_root):
        self.frontier_map = np.load(file_root+'/frontier_map.npy')
        self.room_map = np.load(file_root + '/room_map.npy')

    def load_config(self, cfg_file):
        print("Loading configuration from: %s" % cfg_file)
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f.read())

        return config


    def temp_init(self):
        self.node_idx = 0
        self.obj_locations = [[[tensor(0.6166), 244, 321, 0], [tensor(0.6863), 244, 315, 0]], [[tensor(0.7312), 315, 294, 1]], [], [], [], [[tensor(0.6114), 275, 276, 1]], [], [], [[tensor(0.6166), 345, 270, 1], [tensor(0.7447), 271, 278, 1], [tensor(0.7429), 267, 275, -1], [tensor(0.6965), 277, 278, -1], [tensor(0.6601), 326, 236, 5], [tensor(0.7429), 325, 236, 5], [tensor(0.7403), 327, 235, 5], [tensor(0.6866), 343, 233, 5]], [], [], [[tensor(0.6274), 322, 292, 1], [tensor(0.6184), 275, 262, 1]], [[tensor(0.6653), 314, 269, 1], [tensor(0.6588), 314, 269, 1], [tensor(0.6498), 314, 269, 1], [tensor(0.6478), 315, 272, 1], [tensor(0.6610), 314, 269, 1], [tensor(0.6484), 314, 269, 1], [tensor(0.6295), 315, 269, 1], [tensor(0.6809), 314, 270, 1], [tensor(0.6779), 314, 270, 1], [tensor(0.6127), 314, 270, 1]], [], [[tensor(0.7667), 243, 318, 0], [tensor(0.6498), 344, 256, 1], [tensor(0.7628), 243, 317, 0], [tensor(0.7569), 241, 312, 0]], [], [[tensor(0.6148), 328, 284, 1]], [[tensor(0.7009), 235, 346, 0], [tensor(0.7099), 265, 271, -1], [tensor(0.7320), 328, 253, 1], [tensor(0.7008), 333, 253, 1], [tensor(0.6898), 243, 345, 0], [tensor(0.7174), 234, 345, 0], [tensor(0.7546), 245, 336, 0], [tensor(0.6934), 266, 271, 1], [tensor(0.7207), 269, 269, 1], [tensor(0.7456), 329, 255, 1], [tensor(0.8340), 327, 256, 1], [tensor(0.7551), 267, 274, 1], [tensor(0.7917), 268, 273, -1], [tensor(0.7010), 269, 272, -1]], [], [], [[tensor(0.7197), 233, 328, 0]], [], [[tensor(0.7467), 313, 266, 1], [tensor(0.7534), 314, 270, 1], [tensor(0.7725), 314, 267, 1], [tensor(0.7518), 314, 267, 1]], [], [], [[tensor(0.6180), 345, 270, 1]], [[tensor(0.7291), 322, 293, 1]], [], [], [], [], [], [], [], [[tensor(0.7525), 331, 286, 1]], [], [[tensor(0.7326), 318, 257, 1], [tensor(0.7970), 321, 260, 1], [tensor(0.7383), 319, 258, 1], [tensor(0.8005), 313, 270, 1], [tensor(0.6610), 308, 268, -1]], [[tensor(0.7071), 338, 230, 5], [tensor(0.7434), 338, 230, 5], [tensor(0.7407), 337, 230, 5], [tensor(0.7568), 337, 231, 5]], [[tensor(0.6111), 333, 225, 5]], [], [], [[tensor(0.6920), 260, 311, 0], [tensor(0.6776), 260, 310, 0], [tensor(0.7055), 258, 307, 0]], []]


    def obj_map_update(self, objectMap):
        self.node_idx = 0
        self.obj_locations = objectMap

    def room_map_update(self, roomMap):
        self.room_map = roomMap.cpu().numpy()

    def room_map_update_v2(self, roomMap):
        self.room_map = roomMap

    def obj_map_update_v2(self, objectMap):
        self.node_idx = 0
        self.obj_locations = objectMap


    def create_voxels_bbox(self,coordinates):
        if coordinates.size > 0:
            x_coords, y_coords = zip(*coordinates[:,:2])
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            return (min_x, min_y), (max_x, max_y)
        return None

    def agent_voxel_update(self, voxel):
        self.agent_voxel = voxel

    def cow_room_node_generation(self):

        global_room_info = defaultdict()

        analysis_room_voxels = {}

        for room_index, room_voxels in self.room_map.items():
            global_room_info[self.categories_rooms[room_index]] = list()
            candidates = []
            for room_voxel in room_voxels:
                candidates.append([room_voxel[0][0], room_voxel[0][1]])

            # x_coords = [v[0] for v in candidates]
            # y_coords = [v[1] for v in candidates]
            #
            # plt.figure(figsize=(10, 6))
            # plt.scatter(x_coords, y_coords, alpha=0.6)
            # plt.title('Voxel Locations')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.grid(True)
            # # plt.axhline(0, color='black', linewidth=0.5)
            # # plt.axvline(0, color='black', linewidth=0.5)
            # plt.show()

            dbscan = DBSCAN(eps=15, min_samples=10)
            dbscan.fit(candidates)
            labels = dbscan.labels_

            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)


            for instance_id, label in enumerate(unique_labels):
                self.ins_room_cout += 1
                class_member_mask = (labels == label)
                xy = np.array(candidates)[class_member_mask]



                instance_info = dict()

                global_room_bbox = self.create_voxels_bbox(xy)

                (min_x, min_y), (max_x, max_y) = global_room_bbox[0], global_room_bbox[1]
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                room_insId = f"{self.categories_rooms[room_index]}_{instance_id}"

                analysis_room_voxels[room_insId] = xy


                instance_info['node_idx'] = self.node_idx
                self.node_idx += 1

                instance_info['id'] = room_insId
                instance_info['global_center_pose'] = [center_x, center_y]
                instance_info['composition'] = list()
                instance_info['bbox'] = global_room_bbox
                instance_info['type'] = 'R' #room
                global_room_info[self.categories_rooms[room_index]].append(instance_info)
            # print(f"{room_index} : {global_room_bbox}")

        return global_room_info, analysis_room_voxels

    def visual_room_node(self, global_room_info):
        plt.figure(figsize=(10, 10))
        plt.title("Bounding Box Colors")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.xlim(0, 600)
        plt.ylim(0, 600)
        plt.grid(True)

        # 각 방에 대한 색상 지정
        colors = ['red', 'green', 'blue', 'purple', 'grey', 'olive', 'orangered', 'brown', 'yellow']

        for rooms, color in zip(global_room_info, colors):

            for room in global_room_info[rooms]:
                bbox = room['bbox']
                rect = plt.Rectangle(bbox[0], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], fill=False,
                                     edgecolor=color,
                                     linewidth=2)
                plt.gca().add_patch(rect)

                center = room['global_center_pose']
                plt.scatter(center[0], center[1], marker='x', color='black')

                # 방 ID 표시
                plt.text(center[0], center[1], room['id'], color=color, fontsize=12)

        # 축 설정

        plt.show()



    def find_nearest_room(self,obj_location, global_room_info):

        def calculate_distance(p1, p2):

            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        min_distance = float('inf')
        nearest_room_idx = -1
        semantic_room = ""
        for room_type, rooms in global_room_info.items():
            for r_idx, room in enumerate(rooms):
                room_center = room['global_center_pose']
                distance = calculate_distance(obj_location, room_center)
                if distance < min_distance:
                    min_distance = distance
                    nearest_room_idx = r_idx #room['node_idx']
                    semantic_room = room_type

        return nearest_room_idx , semantic_room

    def object_node_generation(self, global_room_info):
        global_object_info = list()

        [global_object_info.append(dict()) for i in range(self.node_idx)]

        analysis_object_voxels = {}

        for obj_index, obj_voxels in self.obj_locations.items():
            instance_info = dict()
            candidates = []
            obj_confidence = 0.0
            for obj_voxel in obj_voxels:
                candidates.append([obj_voxel[0][0], obj_voxel[0][1]])
                obj_confidence = obj_voxel[0][-1].item()
            obj_category = self.categories_objects[obj_index]

            candidates = np.array(candidates)
            global_obj_bbox = self.create_voxels_bbox(candidates)
            (min_x, min_y), (max_x, max_y) = global_obj_bbox[0], global_obj_bbox[1]
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            # 해당 물체가 어느 방에 속하는지 ?
            nearest_room_idx, room_type = self.find_nearest_room([center_x, center_y], global_room_info)
            if room_type == "":
                continue
            # print(room_type)
            # print(nearest_room_idx)
            obj_room_infos = global_room_info[room_type][nearest_room_idx]

            obj_semantic_pose = obj_room_infos['id']
            global_room_info[room_type][nearest_room_idx]['composition'].append(self.node_idx)

            instance_info['node_idx'] = self.node_idx
            self.node_idx += 1

            instance_info['type'] = self.classifiy_object_type(obj_category)
            instance_info['score'] = obj_confidence

            ins_id = f"{obj_category}_{0}"
            instance_info['id'] = ins_id
            instance_info['global_pose'] = [center_x, center_y]
            instance_info['semantic_pose'] = obj_semantic_pose

            analysis_object_voxels[ins_id] = candidates
            global_object_info.append(instance_info)


        instance_info = dict()
        instance_info['node_idx'] = self.node_idx
        self.node_idx += 1

        instance_info['type'] = "AGENT"
        instance_info['score'] = 0
        instance_info['id'] = 'robot_agent'
        instance_info['global_pose'] = [self.agent_voxel[0], self.agent_voxel[2]]
        instance_info['semantic_pose'] = 0
        global_object_info.append(instance_info)

        return global_room_info, global_object_info, analysis_object_voxels

    def classifiy_object_type(self, object_category):
        # Define item and furniture nodes
        item_nodes = [
        "book", "bottle", "box", "knife", "candle", "cd", "cellphone", "cup", "fork",
        "newspaper", "pencil", "pepper shaker", "plate", "pot", "salt shaker", "statue",
        "tennis racket", "watch", "apple", "baseball bat", "basketball", "bowl", "laptop",
        "mug", "remotecontrol", "spray bottle", "television", "vase", "clock", "garbage can", "plant",
        "lamp", "painting", "pillow"

        ]

        furniture_nodes = [
        "bed", "chair", "desk", "table", "drawer", "dresser",
         "shelf", "sofa", "tv stand"
        ]
        if object_category in item_nodes:
            return "I" #Item
        elif object_category in furniture_nodes:
            return "F" #Furniture
        else:
            return "unknown"

    def edge_generation(self, global_room_info, semantic_object_map):

        edge_index_dict = {'inclusion' : list(),
                           'direction' : list(),
                           'proximity_near' : list(),
                           'proximity_nextTo' : list()}
        all_rooms = [item for sublist in global_room_info.values() for item in sublist]
        for all_room_idx, room_ins in enumerate(all_rooms):


            edge_index_dict['inclusion'].append([room_ins['node_idx'], room_ins['node_idx']]) # self-connection : 포함된 물체가 없을 때를 대비

            for idx in range(len(room_ins['composition'])):
                if idx == len(room_ins['composition']):
                    start_num = idx-1
                else:
                    start_num = idx + 1
                obj_idx = room_ins['composition'][idx]

                obj_info = semantic_object_map[obj_idx]

                edge_index_dict['inclusion'].append([obj_info['node_idx'], room_ins['node_idx']])

                for other_idx in room_ins['composition'][start_num:]:

                    other_obj_info = semantic_object_map[other_idx]

                    #type check , 관계 추론 전 단계 -> type 별로 간선 우선 연결
                    if (obj_info['type'] == 'F') and (other_obj_info['type'] == 'F'):
                        edge_index_dict['direction'].append([obj_info['node_idx'], other_obj_info['node_idx']])

                    if (obj_info['type'] == 'F') and (other_obj_info['type'] == 'I'):
                        edge_index_dict['proximity_near'].append([obj_info['node_idx'], other_obj_info['node_idx']])

                    if (obj_info['type'] == 'I') and (other_obj_info['type'] == 'F'):
                        edge_index_dict['proximity_near'].append([obj_info['node_idx'], other_obj_info['node_idx']])

            #방과 방 간의 proximity_nextTo 간선 연결해야함

            if all_room_idx == len(all_rooms)-1:
                continue
            else:
                room_start = all_room_idx + 1

            for other_room in all_rooms[room_start:]:
                edge_index_dict['proximity_nextTo'].append([all_rooms[all_room_idx]['node_idx'], other_room['node_idx']])



        semantic_object_map[:len(all_rooms)] = all_rooms

        return edge_index_dict, semantic_object_map

    def inclusion_reasoning(self,result, edge_index, semantic_object_map):
        for edges in edge_index:
            subIdx = edges[0]
            objIdx = edges[1]
            result.append([subIdx,self.relation_labels.index('in'), objIdx])
        return result

    def direction_reasoning(self,result, edge_index, semantic_object_map):
        #left, right, behind, front
        for edges in edge_index:
            subIdx = edges[0]
            objIdx = edges[1]
            subject_pos = np.array(semantic_object_map[subIdx]['global_pose'])
            object_pos = np.array(semantic_object_map[objIdx]['global_pose'])

            diff = np.array(object_pos) - np.array(subject_pos)
            if abs(diff[0]) > abs(diff[1]):  # Horizontal difference
                if diff[0] > 0:
                    result.append([subIdx, self.relation_labels.index('left of'), objIdx])
                else:
                    result.append([subIdx, self.relation_labels.index('right of'), objIdx])

            if abs(diff[1]) > abs(diff[0]):
                if diff[1] > 0:
                    result.append([subIdx, self.relation_labels.index('behind'), objIdx])
                else:
                    result.append([subIdx, self.relation_labels.index('front'), objIdx])

        return result

    def near_reasoning(self,result, edge_index, semantic_object_map):
        # near
        near_threshold = 10
        for edges in edge_index:
            subIdx = edges[0]
            objIdx = edges[1]
            subject_pos = np.array(semantic_object_map[subIdx]['global_pose'])
            object_pos = np.array(semantic_object_map[objIdx]['global_pose'])
            dist = np.linalg.norm(subject_pos - object_pos)

            if dist <= near_threshold:
                result.append([subIdx, self.relation_labels.index('near'), objIdx])


        return result

    def calculate_overlap_area(self,bbox1, bbox2):
        """Calculate the overlap area between two bounding boxes."""
        x1_min, y1_min = bbox1[0]
        x1_max, y1_max = bbox1[1]
        x2_min, y2_min = bbox2[0]
        x2_max, y2_max = bbox2[1]

        overlap_width = min(x1_max, x2_max) - max(x1_min, x2_min)
        overlap_height = min(y1_max, y2_max) - max(y1_min, y2_min)

        if overlap_width > 0 and overlap_height > 0:
            return overlap_width * overlap_height
        else:
            return 0

    def plot_overlapping_boxes(self,room1, room2):


        fig, ax = plt.subplots(1)

        box1 = room1['bbox']
        box2 = room2['bbox']
        rect1 = patches.Rectangle((box1[0][0], box1[0][1]), box1[1][0] - box1[0][0], box1[1][1] - box1[0][1],
                                  linewidth=1, edgecolor='r', facecolor='none', label=room1['id'])
        rect2 = patches.Rectangle((box2[0][0], box2[0][1]), box2[1][0] - box2[0][0], box2[1][1] - box2[0][1],
                                  linewidth=1, edgecolor='b', facecolor='none', label=room2['id'])


        min_x_overlap = max(box1[0][0], box2[0][0])
        max_x_overlap = min(box1[1][0], box2[1][0])
        min_y_overlap = max(box1[0][1], box2[0][1])
        max_y_overlap = min(box1[1][1], box2[1][1])

        overlap = 0
        if min_x_overlap < max_x_overlap and min_y_overlap < max_y_overlap:
            overlap = (max_x_overlap - min_x_overlap) * (max_y_overlap - min_y_overlap)

        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.set_xlim(100, 400)
        ax.set_ylim(150, 400)

        plt.text((box1[0][0] + box1[1][0]) / 2, (box1[0][1] + box1[1][1]) / 2, 'Box 1', ha='center', color='r')
        plt.text((box2[0][0] + box2[1][0]) / 2, (box2[0][1] + box2[1][1]) / 2, 'Box 2', ha='center', color='b')



        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Overlapping Bounding Boxes')
        plt.legend()
        plt.show()

    def nextTo_reasoning(self,result, edge_index, semantic_object_map):
        # room-room --> next To
        overlap_threshold = 500
        for edges in edge_index:
            subIdx = edges[0]
            objIdx = edges[1]
            room_1 = semantic_object_map[subIdx]
            room_2 = semantic_object_map[objIdx]

            overlap_area = self.calculate_overlap_area(room_1['bbox'], room_2['bbox'])
            center_distance = np.linalg.norm(
                np.array(room_1['global_center_pose']) - np.array(room_2['global_center_pose']))

            # self.plot_overlapping_boxes(room_1, room_2)

            if overlap_area >= overlap_threshold:
                result.append([subIdx, self.relation_labels.index('next to'), objIdx])



        return result

    def robot_agent_reasoning(self,result, edge_index, semantic_object_map):
        shortest_RoomIdx = -1
        roomIn_objects = list()
        room_elements = sorted(set(sum(edge_index, [])))
        margin = 20
        robot_dist_threshold = 60
        min_distance = float('inf')
        agent_pos = semantic_object_map[-1]['global_pose']
        agent_nodeIdx = semantic_object_map[-1]['node_idx']


        #agent- room relation

        for room in room_elements:
            #bedroom -- agent --> near? or not?

            (min_x, min_y), (max_x, max_y) = semantic_object_map[room]['bbox']

            room_center_pos = semantic_object_map[room]['global_center_pose']

            dist = np.linalg.norm(np.array(agent_pos) - np.array(room_center_pos))

            is_near = (
                    (min_x - margin <= agent_pos[0] <= max_x + margin) and
                    (min_y - margin <= agent_pos[1] <= max_y + margin)
            )
            if is_near:
                if dist < min_distance:
                    min_distance = dist
                    shortest_RoomIdx = semantic_object_map[room]['node_idx']


                    roomIn_objects = semantic_object_map[shortest_RoomIdx]['composition']
                    ###########################
                    # agent - object relation

        if shortest_RoomIdx != -1:
            result.append([agent_nodeIdx, self.relation_labels.index('positioned at'), shortest_RoomIdx])
            for roomIn_objIdx in roomIn_objects:
                obj_Pose = semantic_object_map[roomIn_objIdx]['global_pose']

                dist = np.linalg.norm(np.array(obj_Pose) - np.array(agent_pos))

                if dist <= robot_dist_threshold:
                    diff = np.array(obj_Pose) - np.array(agent_pos)
                    if abs(diff[0]) > abs(diff[1]):  # Horizontal difference
                        if diff[0] > 0:
                            result.append([agent_nodeIdx, self.relation_labels.index('left of'), roomIn_objIdx])
                        else:
                            result.append([agent_nodeIdx, self.relation_labels.index('right of'), roomIn_objIdx])

                    if abs(diff[1]) > abs(diff[0]):
                        if diff[1] > 0:
                            result.append([agent_nodeIdx, self.relation_labels.index('behind'), roomIn_objIdx])
                        else:
                            result.append([agent_nodeIdx, self.relation_labels.index('front'), roomIn_objIdx])
        return result

    def rule_based_relation_extraction(self,edge_index_dict,semantic_object_map):


        relation_dict = dict()
        edge_index_dict['agent_relation'] = list()
        for edge_type in edge_index_dict:

            spatial_relation = list()
            if edge_type == 'inclusion':
                relation_list = self.inclusion_reasoning(spatial_relation ,edge_index_dict[edge_type],
                                                         semantic_object_map)
            elif edge_type == 'direction':
                relation_list = self.direction_reasoning(spatial_relation, edge_index_dict[edge_type],
                                                            semantic_object_map)
            elif edge_type == 'proximity_near':
                relation_list = self.near_reasoning(spatial_relation, edge_index_dict[edge_type],
                                                            semantic_object_map)
            elif edge_type == 'proximity_nextTo':
                relation_list = self.nextTo_reasoning(spatial_relation, edge_index_dict[edge_type],
                                                       semantic_object_map)
            elif edge_type == 'agent_relation':
                relation_list = self.robot_agent_reasoning(spatial_relation,
                                                         edge_index_dict['proximity_nextTo'],
                                                         semantic_object_map)

            relation_dict[edge_type] = np.array(relation_list)

        return relation_dict


    def context_prompt_generation(self,semantic_context_map, spatial_relation, mode):

        room_indexs = np.unique(spatial_relation['inclusion'][:,2])

        #[1] Semantic Spatial Context
        semantic_spatial_context = list()
        for room_idx in room_indexs:

            contain_sentence = ""
            direction_sentence = ""
            near_sentence = ""


            room_info = semantic_context_map[room_idx]
            room_id = room_info['id'] # ex) badroom_0, 1 ..,2
            indices = np.argwhere(spatial_relation['inclusion'][:,2] == room_idx)

            for idx in indices:
                in_triple = spatial_relation['inclusion'][idx][0]
                in_subjectIdx = in_triple[0]
                in_subjectLabel = semantic_context_map[in_subjectIdx]['id']
                contain_sentence += in_subjectLabel + " "

                # 방 안에 물체들마다 direction, proximity_near 관계 추출 후 문장 생성

                # direction relationship

                direction_subIdx = np.argwhere(spatial_relation['direction'][:,0] == in_subjectIdx)
                for direction_idx in direction_subIdx:
                    direction_triple = spatial_relation['direction'][direction_idx][0]
                    direction_objLabel = semantic_context_map[direction_triple[2]]['id']
                    direction_relLabel = self.relation_labels[direction_triple[1]]
                    direction_sentence += f'The {in_subjectLabel} is to the {direction_relLabel} {direction_objLabel}, '

                # proximity_near relationship

                near_subIdx = np.argwhere(spatial_relation['proximity_near'][:,0] == in_subjectIdx)
                for near_idx in near_subIdx:
                    near_triple = spatial_relation['proximity_near'][near_idx][0]
                    near_objLabel = semantic_context_map[near_triple[2]]['id']
                    near_relLabel = self.relation_labels[near_triple[1]]
                    near_sentence += f'The {in_subjectLabel} is to the {near_relLabel} {near_objLabel}, '



            sentence = f"{room_id} : contains {contain_sentence}. {direction_sentence}. {near_sentence}. "

            semantic_spatial_context.append(sentence)


        #Room-Room Connection Relationship
        nextTo_sentence = ""
        for nextTo_triple in spatial_relation['proximity_nextTo']:
            subRoomIdx = nextTo_triple[0]
            subRoomLabel = semantic_context_map[subRoomIdx]['id']
            nextTo_relLabel = self.relation_labels[nextTo_triple[1]]
            objRoomIdx = nextTo_triple[2]
            objRoomLabel = semantic_context_map[objRoomIdx]['id']
            nextTo_sentence += f'The {subRoomLabel} is {nextTo_relLabel} the {objRoomLabel}. '
        semantic_spatial_context.append(nextTo_sentence)

        #######################################################################################
        # [2] Agent Location Context
        agent_location_context = list()
        agent_sentence = ""
        for agent_triple in spatial_relation['agent_relation']:
            agentIdx = agent_triple[0]
            agentLabel = semantic_context_map[agentIdx]['id']
            relLabel = self.relation_labels[agent_triple[1]]
            objectIdx = agent_triple[2]
            objectLabel = semantic_context_map[objectIdx]['id']
            agent_sentence += f'The {agentLabel} is {relLabel} the {objectLabel}. '
        agent_location_context.append(agent_sentence)

        return semantic_spatial_context, agent_location_context

    def start_process(self,goal_object):

        # 2024.03.04 : Semantic Object Mapping 1차 구현 * global_room_info, global_obj_info

        global_room_info, analysis_room_voxels = self.cow_room_node_generation()


        global_room_info, semantic_object_map, analysis_obj_voxels = self.object_node_generation(global_room_info)

        edge_index_dict, semantic_context_map = self.edge_generation(global_room_info, semantic_object_map)

        # Rule-based Relation Extraction
        spatial_relations = self.rule_based_relation_extraction(edge_index_dict,semantic_object_map)

        # # Context-based LLM Prompting Part
        # [1] Context Prompt Generation

        obj_proba, room_proba = self.llm.prompting(goal_object, semantic_context_map, spatial_relations,
                                                   self.relation_labels, self.experiments_config)

        return obj_proba, room_proba


        # frontier_areas = self.frontier_split_area(self.frontier_locations_12)
        # area_info = self.select_frontier_room(frontier_areas)
        # area_obj_info, max_confidence_objects = self.select_area_objects(area_info)
        #
        # promptInfo = {}
        #
        # area_triple_rel = self.spatial_relation_reasoning(area_obj_info, max_confidence_objects)
        #
        # robot_object_rel, robot_room_description = self.agent_centric_description(area_obj_info)
        #
        # promptInfo['area_triple_rel'] = area_triple_rel
        # promptInfo['robot_object_rel'] = robot_object_rel
        # promptInfo['robot_room_description'] = robot_room_description
        # promptInfo['area_obj_info'] = area_obj_info
        #
        #
        #
        #
        # # self.print_triple_rel(area_triple_rel)
        #
        #
        # self.frontier_room_object_plot(frontier_areas, area_obj_info, max_confidence_objects)
        # self.frontier_room_plot(frontier_areas, area_info)
        # self.plot_frontier_areas(frontier_areas)
        # # sys.exit()
        #
        # merge_room = np.zeros((600, 600), dtype=int)
        # object_semantic_room = np.zeros((600, 600), dtype=int)
        #
        # for obj_idx in range(len(self.obj_locations)):
        #     obj_num = len(self.obj_locations[obj_idx])
        #     if obj_num == 0:
        #         continue
        #
        #     obj_location_mtx = np.array(self.obj_locations[obj_idx])[:, 1:]
        #     obj_confidence_mtx = np.array(self.obj_locations[obj_idx])[:, 0]
        #     high_obj_index = np.argmax(obj_confidence_mtx)
        #
        #     obj_gPose = obj_location_mtx[high_obj_index] #[,2]
        #     # obj_gBbox = obj_bbox_mtselfx[high_obj_index][0] #[4,2]
        #
        #     obj_x = int(obj_gPose[0])
        #     obj_y = int(obj_gPose[1])
        #
        #     object_semantic_room[obj_x, obj_y] = obj_idx + 1
        #     # for corner in obj_gBbox:
        #     #     c_x = corner[0]
        #     #     c_y = corner[1]
        #     #     object_semantic_room[c_x, c_y] = obj_idx
        #
        #
        # # 각 방을 순회하면서 신뢰 점수가 0.65 이상인 셀의 인덱스 추출
        # for room_type_index in range(self.room_map.shape[1]):
        #     room_map = self.room_map[0, room_type_index, :, :]
        #     high_confidence_cells = room_map >= 0.65
        #     merge_room[high_confidence_cells] = room_type_index + 1
        #
        #
        #
        # if self.visualization:
        #     self.merge_room_visualization(merge_room)
        #     self.frontier_room_visualization(self.frontier_map)
        #     self.object_semantic_map_visualization(object_semantic_room)
        #
        # #Frontier Map
        # frontier_ = self.frontier_map > 0
        # print(frontier_)


    def start_process_v2(self,goal_object):

        # 2024.03.04 : Semantic Object Mapping 1차 구현 * global_room_info, global_obj_info

        global_room_info, analysis_room_voxels = self.cow_room_node_generation()

        # 시각화
        # self.visual_room_node(global_room_info)


        global_room_info, semantic_object_map , analysis_object_voxels = self.object_node_generation(global_room_info)

        edge_index_dict, semantic_context_map = self.edge_generation(global_room_info, semantic_object_map)

        # Rule-based Relation Extraction
        spatial_relations = self.rule_based_relation_extraction(edge_index_dict,semantic_object_map)

        # # Context-based LLM Prompting Part
        # [1] Context Prompt Generation
        mode = 'sentence'

        obj_proba, room_proba = self.llm.prompting(goal_object, semantic_context_map, spatial_relations,
                                                   self.relation_labels, self.experiments_config)

        return obj_proba, room_proba, analysis_room_voxels, analysis_object_voxels, self.experiments_config



if __name__ == '__main__':
    CMG_module = CMGModule()
    start = [300,300]
    CMG_module.agent_info_update(start)
    prob_array_obj, prob_array_room = CMG_module.start_process('Alarm Clock on a dresser')




    # def load_npy(self,file_root):
    #     self.frontier_map = np.load(file_root+'/frontier_map.npy')
    #     self.room_map = np.load(file_root + '/room_map.npy')
#
#     def merge_room_visualization(self, merge_room):
#         import matplotlib.pyplot as plt
#         import matplotlib.colors as mcolors
#
#         cmap = plt.cm.get_cmap('tab10', 10)
#         new_colors = cmap(np.linspace(0, 1, 10))
#         new_colors[0, :] = np.array([1, 1, 1, 1])  # 0번 색상을 흰색으로 설정
#         new_cmap = mcolors.ListedColormap(new_colors)
#
#         plt.figure(figsize=(10, 10))
#         plt.imshow(merge_room, cmap=new_cmap, vmin=0, vmax=9)
#         # plt.colorbar(label='Room Type')
#
#         cbar = plt.colorbar(label='Room Type', ticks=range(10))
#         cbar.ax.set_yticklabels(['None'] + self.rooms)
#
#         plt.title("Visualization of Merged Room Map")
#         plt.xlabel("X-coordinate")
#         plt.ylabel("Y-coordinate")
#         plt.show()
#
#     def frontier_room_visualization(self, frontier):
#         import matplotlib.pyplot as plt
#         binary_frontier_map = frontier.astype(int)
#         plt.figure(figsize=(8, 8))
#         plt.imshow(binary_frontier_map, cmap='binary', interpolation='none')
#         plt.gca().invert_yaxis()
#         plt.title("Frontier Map")
#         plt.xlabel("X-coordinate")
#         plt.ylabel("Y-coordinate")
#         plt.show()
#
#     def object_semantic_map_visualization(self,object_semantic_map):
#         import matplotlib.pyplot as plt
#         from matplotlib.colors import ListedColormap
#         # Generate a color map with a unique color for each category
#         colors = plt.cm.get_cmap('tab20', len(self.categories_21))  # Using 'tab20' colormap for variety
#
#         cmap = ListedColormap(colors(np.linspace(0, 1, len(self.categories_21))))
#         # x = np.where(object_semantic_map == 41)
#         # Plotting
#
#         plt.figure(figsize=(12, 12))
#         plt.imshow(object_semantic_map, cmap=cmap)
#         # plt.imshow(object_semantic_map)
#         # Creating a color legend with labels
#         handles = [plt.Rectangle((0, 0), 1, 1, color=colors(i)) for i in range(len(self.categories_21))]
#         plt.legend(handles, self.categories_21, bbox_to_anchor=(1.05, 1), loc='upper left')
#
#         plt.title('Object Semantic Map Visualization')
#         plt.xlabel('X-axis')
#         plt.ylabel('Y-axis')
#         plt.show()
#
#
#     def plot_frontier_sub_room_map(self,sub_room_map, room_label):
#         import matplotlib.pyplot as plt
#
#         plt.figure(figsize=(8, 8))
#         plt.scatter(sub_room_map[0], sub_room_map[1], label=room_label)
#         # 그래프 설정
#
#         Robot_Label = "Robot"
#         plt.scatter(self.robot_locations[0],self.robot_locations[1], marker='x', label=f'{Robot_Label}')
#
#         plt.xlabel('X Coordinate')
#         plt.ylabel('Y Coordinate')
#         plt.title('Frontier Sub Room Map')
#         plt.xlim(0, 600)
#         plt.ylim(0, 600)
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#     def plot_frontier_areas(self,areas):
#         import matplotlib.pyplot as plt
#
#         plt.figure(figsize=(8, 8))
#
#         # # 각 영역에 대해 시각화
#         for area_key in areas:
#             area_data = areas[area_key]
#             plt.scatter(area_data[:, 0], area_data[:, 1], label=area_key)
#
#         Robot_Label = "Robot"
#         plt.scatter(self.robot_locations[0],self.robot_locations[1], marker='x', label=f'{Robot_Label}')
#         plt.grid(True)
#         # 그래프 설정
#         plt.xlabel('X Coordinate')
#         plt.ylabel('Y Coordinate')
#         plt.title('Frontier Areas')
#         plt.xlim(0, 600)
#         plt.ylim(0, 600)
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#     def frontier_room_object_plot(self,frontier_areas, area_obj_info, max_confidence_objects):
#         import matplotlib.pyplot as plt
#         # 시각화를 위해 각 Area별로 subplot 생성
#         fig, axes = plt.subplots(1, len(area_obj_info), figsize=(15, 5))
#
#         for ax, (area, data) in zip(axes, area_obj_info.items()):
#             ax.set_title(area)
#             ax.set_xlim(0, 600)
#             ax.set_ylim(0, 600)
#             Robot_Label = "Robot"
#             ax.scatter(self.robot_locations[0], self.robot_locations[1], marker='x', label=f'{Robot_Label}')
#             ax.grid(True)
#
#             # Area 내의 방 좌표 시각화
#             for room, coords in data.items():
#
#                 if not room.endswith('objects'):
#                     x, y = zip(*coords)
#                     ax.scatter(x, y, label=f'{room}')
#
#                 # Area 내의 물체 중심점 좌표 시각화
#                 # obj_key = 'objects'
#                 if room == 'objects':
#                     obj_indices = data[room]
#                     for idx in obj_indices:
#                         obj_coord = max_confidence_objects[idx]
#                         ax.scatter(obj_coord[0], obj_coord[1], marker='x', label=f'{self.categories_21[idx]} ({idx})')
#
#                 ax.legend()
#
#
#
#         plt.tight_layout()
#         plt.show()
#
#     def frontier_room_plot(self,frontier_areas,area_info ):
#         import matplotlib.pyplot as plt
#         fig, axes = plt.subplots(1, len(area_info), figsize=(15, 5))
#         if len(area_info) == 1:
#             axes = [axes]
#         for ax, (area, rooms) in zip(axes, area_info.items()):
#             ax.set_title(area)
#             ax.set_xlim(0, 600)
#             ax.set_ylim(0, 600)
#             Robot_Label = "Robot"
#             ax.scatter(self.robot_locations[0], self.robot_locations[1], marker='x', label=f'{Robot_Label}')
#             ax.grid(True)
#
#             # Plot rooms
#             for room, coords in rooms.items():
#                 if coords:
#                     x, y = zip(*coords)
#                     ax.scatter(x, y, label=room)
#
#             # Plot frontier areas
#             if area in frontier_areas and frontier_areas[area].size > 0:
#                 frontier_coords = frontier_areas[area]
#                 ax.scatter(frontier_coords[:, 0], frontier_coords[:, 1], color='red', label='Frontier')
#
#             ax.legend()
#
#
#
#         plt.tight_layout()
#         plt.show()
#
#
#     def frontier_split_area(self, frontier_location_12):
#         from sklearn.cluster import DBSCAN
#         # DBSCAN 클러스터링 수행
#         dbscan = DBSCAN(eps=24, min_samples=3) #eps = 12(0.6m), eps = 24(1.2m)
#         # eps: 인접 셀 간 최대 거리, min_samples: 클러스터를 형성하는 최소 셀 수
#         clusters = dbscan.fit_predict(frontier_location_12)
#         num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
#         print(f"Number of clusters: {num_clusters}")
#
#         areas = {}
#         for i in range(num_clusters):
#             area_key = f"Area{i+1}"
#             areas[area_key] = frontier_location_12[clusters == i]
#
#         return areas
#
#     def select_frontier_room(self, frontier_areas):
#
#         area_info = dict()
#
#         for area_key in frontier_areas:
#
#             max_confidence = 1  # 최고 신뢰도 점수 초기화
#             best_sub_room_info = defaultdict(set)  # 가장 높은 신뢰도를 가진 방의 정보 저장
#
#             for i, loc in enumerate(frontier_areas[area_key]):
#                 x_start = max(0, loc[0] - 12)
#                 y_start = max(0, loc[1] - 12)
#                 sub_room_map = self.room_map[0, :, max(0, loc[0] - 12):min(self.map_size - 1, loc[0] + 13),
#                                max(0, loc[1] - 12):min(self.map_size - 1, loc[1] + 13)]
#
#                 max_x_axis = np.max(sub_room_map, 1)
#                 sub_room_conf = np.max(max_x_axis, 1)
#
#                 if sub_room_conf.max() < self.room_select_threshold:
#                     continue
#
#                 sub_room_max = np.argmax(sub_room_conf, 0)
#                 sub_room_conf = sub_room_conf[sub_room_max]
#
#                 if sub_room_conf <= max_confidence:
#                     max_confidence = sub_room_conf  # 최고 신뢰도 점수 갱신
#                     sub_room_label = self.rooms[sub_room_max]
#                     sub_room = sub_room_map[sub_room_max]
#                     sub_x_axis, sub_y_axis = np.where(sub_room >= self.room_select_threshold)
#
#                     map_room_x = sub_x_axis + x_start
#                     map_room_y = sub_y_axis + y_start
#                     sub_room_map_pos = np.vstack((map_room_x, map_room_y)).T
#
#                       # 이전 방 정보 초기화
#                     for pos in sub_room_map_pos:
#                         best_sub_room_info[sub_room_label].add(tuple(pos))
#                 else:
#                     # sub_room_label = self.rooms[sub_room_max]
#                     old_room_label = list(best_sub_room_info.keys())[0]
#                     best_sub_room_info.pop(old_room_label,None)
#                     max_confidence = sub_room_conf
#
#             area_info[area_key] = best_sub_room_info
#
#         return area_info
#
#     def select_area_objects(self,area_info):
#
#         max_confidence_locations = np.zeros((len(self.categories_21), 2))
#
#         for i, locations in enumerate(self.obj_locations):
#             if locations:
#                 # 신뢰도 점수에 따라 정렬
#                 sorted_locations = sorted(locations, key=lambda x: x[0], reverse=True)
#                 # 최대 신뢰도 점수를 가진 위치 저장
#                 max_confidence_locations[i] = [sorted_locations[0][1], sorted_locations[0][2]]
#         area_info_cp = copy.deepcopy(area_info)
#         for area, area_room in area_info.items():
#             for room, coords in area_room.items():
#                 temp_obj = set()
#                 if coords:
#                     for coord in coords:
#                         room_coord = np.tile(list(coord), (len(self.categories_21), 1))
#                         dist_room_obj = np.square(room_coord - max_confidence_locations)
#                         dist_room_obj = np.sqrt(np.sum(dist_room_obj, axis=1))
#                         near_room_obj = np.where(dist_room_obj < 24)[0]
#                         [temp_obj.add(i) for i in near_room_obj]
#
#                 new_key = "objects"
#                 area_info_cp[area][new_key] = list(temp_obj)
#         return area_info_cp, max_confidence_locations
#
#     def spatial_relation_reasoning(self,area_obj_info, max_confidence_objects):
#         x_threshold = 30
#         y_threshold = 30
#         triple_relation = {}
#         for area, data in area_obj_info.items():
#             obj_room_key = list(data.keys())[1]
#
#             obj_indices = data[obj_room_key]
#             n_objects = len(obj_indices)
#             obj_center_list = max_confidence_objects[obj_indices]
#
#             distances = np.zeros((n_objects, n_objects))
#             directions = np.zeros((n_objects, n_objects, 2))
#
#             area_triplet_relation = []
#
#             for sub_idx in range(n_objects):
#                 for obj_idx in range(sub_idx+1, n_objects):
#                     if sub_idx == obj_idx:
#                         continue
#
#                     diff = obj_center_list[obj_idx] - obj_center_list[sub_idx] #물체 간의 상대적 위치 벡터
#                     distances[sub_idx, obj_idx] = np.linalg.norm(diff) #벡터의 길이(물체 간 거리)
#                     directions[sub_idx, obj_idx] = diff / (np.linalg.norm(diff) + 1e-10)  # 방향은 단위 벡터
#
#                     subject_index = obj_indices[sub_idx]
#                     object_index = obj_indices[obj_idx]
#                     if distances[sub_idx,obj_idx] < 15:
#                         nf_relation = 'near'
#                         area_triplet_relation.append([subject_index, object_index, self.spatial_relations.index(nf_relation)])
#
#
#                     x_diff_abs = abs(diff[0])
#                     y_diff_abs = abs(diff[1])
#
#                     # 복합 관계 추론
#                     if x_diff_abs > x_threshold or y_diff_abs > y_threshold:
#                          # 초기 복합 관계 설정
#                         if directions[sub_idx, obj_idx][0] > 0:
#                             if directions[sub_idx, obj_idx][1] > 0:
#                                 combined_relation = 'northeast'
#                             else:
#                                 combined_relation = 'southeast'
#                         else:
#                             if directions[sub_idx, obj_idx][1] > 0:
#                                 combined_relation = 'northwest'
#                             else:
#                                 combined_relation = 'southwest'
#                         area_triplet_relation.append(
#                             [subject_index, object_index, self.spatial_relations.index(combined_relation)])
#                     else:
#                         # 기본 방향성 할당
#
#                         # if x_diff_abs <= x_threshold:
#                         #     fb_relation = 'infront' if directions[sub_idx, obj_idx][1] > 0 else 'behind'
#                         #     area_triplet_relation.append(
#                         #         [subject_index, object_index, self.spatial_relations.index(fb_relation)])
#
#                         # if y_diff_abs <= y_threshold:
#                         lr_relation = 'right' if directions[sub_idx, obj_idx][0] > 0 else 'left'
#                         area_triplet_relation.append(
#                             [subject_index, object_index, self.spatial_relations.index(lr_relation)])
#
#
#                         #
#                         # if x_diff_abs <= x_threshold:
#                         #     lr_relation = 'right' if directions[sub_idx, obj_idx][0] > 0 else 'left'
#                         #     area_triplet_relation.append(
#                         #         [subject_index, object_index, self.spatial_relations.index(lr_relation)])
#                         # if y_diff_abs <= y_threshold:
#                         #     fb_relation = 'infront' if directions[sub_idx, obj_idx][1] > 0 else 'behind'
#                         #     area_triplet_relation.append(
#                         #         [subject_index, object_index, self.spatial_relations.index(fb_relation)])
#
#             triple_relation[area] = area_triplet_relation
#
#         return triple_relation
#
#     def print_triple_rel(self, triple_relation):
#         for area, relation in triple_relation.items():
#             print(f"\t =={area}== \t")
#             for triple in relation:
#                 subject_name = self.categories_21[triple[0]]
#                 object_name = self.categories_21[triple[1]]
#                 rel = self.spatial_relations[triple[2]]
#
#                 print(f"\t {subject_name}=={rel}=={object_name} \t")
#             print("\n")
#
#
#     def agent_centric_description(self,area_obj_info):
#
#
#
#         #[1] 로봇의 현재 좌표를 기준으로 가장 근접한 Area 및 방을 찾기
#         dist_room_robot_area = []
#         for area_name, area_data in area_obj_info.items():
#             for room_data in area_data:
#                 if not room_data.endswith('objects'):
#                     room_coord = np.array(list(area_data[room_data]))
#                     robot_coord = np.tile(np.array(self.robot_locations), (len(room_coord), 1))
#                     dist_room_robot = np.square(robot_coord -room_coord)
#                     dist_room_robot = np.sqrt(np.sum(dist_room_robot, axis=1))
#
#                     min_dist_room_robot = np.min(dist_room_robot, 0)
#                     dist_room_robot_area.append(min_dist_room_robot)
#
#         #가장 가까운 방은 물체들이 안나옴 -> 이 경우엔 임의로 가장 먼 방 뽑음
#         robot_near_area_idx = np.argmax(dist_room_robot_area)
#
#         #[2] 로봇과 가장 가까이 있는 Area에 존재하는 물체들 선택
#         robot_near_area = list(area_obj_info.values())[robot_near_area_idx]
#         robot_near_objects = robot_near_area['objects']
#
#         distances = np.zeros((1, len(robot_near_objects)))
#         directions = np.zeros((1, len(robot_near_objects), 2))
#
#         robot_object_relation = []
#         robot_room_relation = f'The robot is located adjacent to the {self.rooms[robot_near_area_idx]}.'
#         # robot_centric_relation.append([0, object_index, self.spatial_relations.index(nf_relation)])
#
#         #[3] 로봇과 물체간의 공간 관계 판단
#         for temp_obj_idx, near_obj_idx in enumerate(robot_near_objects):
#             diff = np.array(self.robot_locations) - np.array(self.obj_locations[near_obj_idx][0][1:]) #center_list[sub_idx]  # 물체 간의 상대적 위치 벡터
#
#             subject_index = 0  # robot
#             object_index = near_obj_idx  # object
#
#             distances[subject_index, temp_obj_idx] = np.linalg.norm(diff)
#             directions[subject_index, temp_obj_idx] = diff / (np.linalg.norm(diff) + 1e-10)  # 방향은 단위 벡터
#             #
#
#
#             nf_relation = 'near' if distances[subject_index, temp_obj_idx] < 25 else 'far'
#             robot_object_relation.append([subject_index, object_index, self.spatial_relations.index(nf_relation)])
#
#             # x_diff_abs = abs(diff[0])
#             # y_diff_abs = abs(diff[1])
#
#
#         return robot_object_relation, robot_room_relation
#         # for area_obj in robot_near_area
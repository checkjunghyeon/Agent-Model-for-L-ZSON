import os
import io
import http.client
import sys
import typing
import urllib.request
import base64
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import re
import numpy as np


class CLPModule:
    # Context-based LLM Prompting
    def __init__(self, categories_objects, categories_rooms, llm_default="Llama2"):

        self.llm_default = llm_default

        self.init_llm()
        self.categories_objects = categories_objects
        self.categories_rooms = categories_rooms

    def init_llm(self):
        if self.llm_default == 'Llama2':
            from src.models.prompting.LLM_Llama2_7b import FastLlama2
            from fastchat.Prompt import PromptGenerator

            self.llm_Llama2 = FastLlama2()
            self.system_promptGene = PromptGenerator()

        elif self.llm_default == 'vertexAI':
            import vertexai
            from vertexai.preview.generative_models import GenerativeModel, Image
            from vertexai.language_models import TextGenerationModel

            os.environ[
                "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/baebro/Downloads/robothor_cp_hojun/robothor_cp/robothor_cp/GOOGLE_KEY/silent-presence-413104-ce55ee8d4b5f.json"

            location = "asia-northeast3"
            project_id = "silent-presence-413104"
            vertexai.init(project=project_id, location=location)

    def load_and_encode_images(self, prompts):
        encoded_prompts = []
        prompt_details = []
        image_count = 0  # 이미지 개수 카운터

        for item in prompts:
            prompt_detail = {"prompt": item}

            if item.startswith('http'):  # 이미지 URL인 경우
                response = typing.cast(http.client.HTTPResponse, urllib.request.urlopen(item))
                image_data = response.read()
                pil_image = PILImage.open(io.BytesIO(image_data))
                encoded_image = Image.from_bytes(image_data)
                encoded_prompts.append(encoded_image)
                prompt_detail["type"] = "image"
                prompt_detail["pil_image"] = pil_image
                image_count += 1
            elif item.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:  # 파일 경로인 경우
                with open(item, "rb") as image_file:
                    image_data = image_file.read()
                    pil_image = PILImage.open(io.BytesIO(image_data))
                    encoded_image = Image.from_bytes(image_data)
                    encoded_prompts.append(encoded_image)
                    prompt_detail["type"] = "image"
                    prompt_detail["pil_image"] = pil_image
                    image_count += 1
            else:  # 텍스트인 경우
                encoded_prompts.append(item)
                prompt_detail["type"] = "text"

            prompt_details.append(prompt_detail)

        return encoded_prompts, prompt_details, image_count

    def display_response(self, prompt_details, response_text):
        # 이미지와 텍스트 출력
        for prompt_detail in prompt_details:
            if prompt_detail["type"] == "image":
                plt.imshow(prompt_detail["pil_image"])
                plt.axis('off')
                plt.show()
            print("Prompt: {} (type: {})".format(prompt_detail['prompt'], prompt_detail['type']))

        for candidate in response_text.candidates:
            for part in candidate.content.parts:
                print("Gemini:\t {}".format(part.text))

    def chat_gemini(self, prompts):
        encoded_prompts, pil_images, image_count = self.load_and_encode_images(prompts)

        # 이미지 개수에 따라 모델 선택
        if image_count > 0:
            model_name = "gemini-pro-vision"
        else:
            model_name = "gemini-pro"

        # GenerativeModel 객체 생성 및 콘텐츠 생성
        model = GenerativeModel(model_name)
        response = model.generate_content(encoded_prompts)

        # print(f"Loaded {image_count} images using the '{model_name}' model.")
        # self.display_response(pil_images, response)
        return response.text

    def chat_plam2(self, prompts):
        encoded_prompts, pil_images, image_count = self.load_and_encode_images(prompts)

        parameters = {
            "temperature": 0.5,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 128,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,
            # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 1,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(
            encoded_prompts[0],
            **parameters,
        )
        return response.text

    def chat_llama2(self, prompts):
        prompt = self.system_promptGene.user_prompt_generation(prompts)
        output = self.llm_Llama2.generate_stream(prompt)
        return output['text']

    def instruction_configuration_for_Llama2(self, seq, goal_object, room_object, room_room, agent_context,
                                             experiments_config ,selected_objects=None):

        total_sentences = "## Semantic Spatial Context ## \n"
        for room in room_object:

            contain_objects = ", ".join(room_object[room]['items'])

            room_sentence = f"\t -{room} contains {contain_objects}. "

            if experiments_config['prompting']['context'] == 'spatial_relation':
                if 'relations' in room_object[room] and experiments_config['context_map']['configure']=='relation':
                    room_sentence += ''
                    for object, relations in room_object[room]['relations'].items():
                        obj_relations = " and ".join(relations)
                        room_sentence += f"{object} is {obj_relations}."
                else:
                    room_sentence += ''
            else:
                room_room = ""
                agent_context[-1] = ""


            if experiments_config['context_map']['configure'] == 'object':
                room_room = ""
                agent_context[-1] = ""


            total_sentences += room_sentence + "\n"

        total_sentences += f"\t -Rooms Proximities : {room_room}\n"
        total_sentences += f"## Agent Location Context ## \n \t-{agent_context[-1]}\n"

        if seq == 1:
            question = (
                f"Given the detailed Semantic Spatial Context of each room, along with the Agent Location Context, "
                f"Consider general patterns of where an '{goal_object}' are usually located in a household, "
                f"and make an inference based on the provided spatial layout and contents of each room. "
                f"Use your AI capabilities to analyze the context and generate these probabilities. "
                f"please estimate the likelihood of finding an '{goal_object}' in each room.")

            format_ = ""
            for idx, room_name in enumerate(list(room_object.keys())):
                format_ += f"-{room_name} : [Probability as a percentage] \n"

            question += f" Don't add any additional explanation, just make sure it fits the format. For your response, please use the following format : \n {format_}."

        elif seq == 2:
            question = (
                f"Given the detailed Semantic Spatial Context of each room, along with the Agent Location Context, "
                f"provide specific probability estimations for the likelihood of finding a '{goal_object}' in relation to the following objects:{selected_objects}."
                f"Use your AI capabilities to analyze the context and generate these probabilities. "
                f"Please provide a percentage value for each object based on the likelihood of finding the '{goal_object}' nearby, considering the room layouts and object placements."
            )

            format_ = ""
            for object in selected_objects:
                format_ += f"-{object} : [Probability as a percentage] \n"

            question += f"\n Don't add any additional explanation. For your response, please use the following format: : \n {format_}"

        return total_sentences + question

    def instruction_configuration_for_VertexAI(self, seq, goal_object, room_object, room_room, agent_context,
                                               selected_objects=None):

        total_sentences = "## Semantic Spatial Context ## \n"
        for room in room_object:

            contain_objects = ", ".join(room_object[room]['items'])

            room_sentence = f"\t -{room} contains {contain_objects}. "
            if 'relations' in room_object[room]:
                room_sentence += 'And '
                for object, relations in room_object[room]['relations'].items():
                    obj_relations = " and ".join(relations)
                    room_sentence += f"{object} is {obj_relations}."
            else:
                room_sentence += ''

            total_sentences += room_sentence + "\n"

        total_sentences += f"\t -Rooms Proximities : {room_room}\n"
        total_sentences += f"## Agent Location Context ## \n \t-{agent_context[0]}\n"

        if seq == 1:
            question = (
                f"Given the detailed Semantic Spatial Context of each room, along with the Agent Location Context, "
                f"Consider general patterns of where an {goal_object} are usually located in a household, "
                f"and make an inference based on the provided spatial layout and contents of each room. "
                f"please estimate the likelihood of finding an '{goal_object}' in each room. Consider general patterns.")

            format_ = ""
            for idx, room_name in enumerate(list(room_object.keys())):
                format_ += f"-{room_name} : [Probability as a percentage] \n"

            question += f"For your response, please use the following format : \n {format_}"

        elif seq == 2:
            question = (
                f"Given the detailed Semantic Spatial Context of each room, along with the Agent Location Context, "
                f"Based on the Semantic Spatial Context and Agent Location Context, can you provide probability estimations for the likelihood of finding an '{goal_object}' in relation to the following objects : {selected_objects}?"
            )

            format_ = ""
            for object in selected_objects:
                format_ += f"-{object} : [Probability as a percentage] \n"

            question += f"\n For your response, please use the following format : \n {format_}"

        return total_sentences + question

    # Improved function to parse the input list into a structured format
    def parse_rooms_improved(self, semantic_context):
        rooms = {}

        for room_description in semantic_context:

            if len(room_description) <= 4:
                continue

            # Check if the description is about room-to-room relationships
            if room_description.startswith("The "):
                relations = room_description.split(". ")
                for relation in relations:
                    if relation:
                        parts = relation.split(" ")
                        if len(parts) >= 3:
                            room1, relation_desc, room2 = parts[0], parts[1], parts[-1]
            else:
                # Split the room description into parts
                parts = room_description.split(" : ")
                room_name = parts[0]
                details = parts[1] if len(parts) > 1 else ""

                # Separate items and relationships
                items, relations = details.split(". ", 1)
                items = items.replace("contains ", "").split(" ")
                rooms[room_name] = {"items": items}

                # Parse spatial relationships
                relations = relations.split(", ")
                for relation in relations:
                    if relation:
                        relation_parts = relation.split(" is to the ")
                        if len(relation_parts) == 2:
                            item, relation_desc = relation_parts
                            rooms[room_name].setdefault("relations", {}).setdefault(item, []).append(relation_desc)

        return rooms, semantic_context[-1]

    def room_parse_llm_response_for_VertexAI(self, response):
        response_dict = {}
        for pair in response.split('-'):
            if pair.strip():
                # Split by ':' and extract the first two elements (room and score)
                parts = pair.split(':')
                if len(parts) >= 2:
                    room, score = parts[0], parts[1]
                    # Remove special characters but keep spaces and underscores
                    room_clean = ''.join(char for char in room if char.isalnum() or char in [' ', '_']).strip()
                    # Replace double spaces (if any) with a single space
                    room_clean = ' '.join(room_clean.split())
                    # Extract numeric part from score
                    score_numeric = ''.join(filter(str.isdigit, score))
                    if score_numeric:  # Check if score_numeric is not empty
                        # Convert score to a percentage and store in the dictionary
                        response_dict[room_clean] = int(score_numeric) / 100
        return response_dict

    def room_parse_llm_response_for_Llama2(self, response):
        response_dict = {}
        matches = re.findall(r'([a-zA-Z ]+\\_0).*?(\d+%|Less than \d+%)', response, re.IGNORECASE)
        for room, prob in matches:
            # 'Less than 1%'의 경우 1% 미만으로 처리
            if 'less than' in prob.lower():
                prob = '1'
            # 확률에서 '%' 문자 제거하고 소수로 변환
            prob_value = float(prob.strip('%')) / 100
            room_cleaned = room.replace('\\', '').strip()
            if len(room_cleaned) >= 15:
                continue
            response_dict[room_cleaned.lower()] = prob_value

        return response_dict

    def object_parse_llm_response_for_Llama2(self, response):
        response_dict = {}
        # 정규 표현식을 사용하여 객체 이름과 확률 추출
        # 'Less than 1%' 또는 '<1%' 표현 처리
        matches = re.findall(r'(\w+)\s*:?[\s\[]*(\d+%|Less than \d+%|<\d+%|\[Probability as a percentage\])', response,
                             re.IGNORECASE)

        for obj, prob in matches:
            if 'less than' in prob.lower() or '<' in prob:
                prob_value = 0.01  # 'Less than 1%' 또는 '<1%'의 경우 1% 미만으로 처리
            elif '[probability as a percentage]' in prob.lower():
                prob_value = None  # 'Probability as a percentage'의 경우 확률이 정해지지 않음
            else:
                prob_value = float(prob.strip('%')) / 100  # 확률에서 '%' 문자 제거하고 소수로 변환
            response_dict[obj] = prob_value
        return response_dict

    def object_parse_llm_response_for_VertexAI(self, response):
        response_dict = {}
        for pair in response.split('-'):
            if pair.strip():

                object_name, score = pair.split(':')
                score = score.replace('%', '').strip()
                if score and score.isdigit():
                    score = int(score) / 100  # Convert score to decimal
                    response_dict[object_name.strip()] = score
                else:
                    response_dict[object_name.strip()] = 0.0

        return response_dict

    def prompting(self, goal_object, semantic_context_map, spatial_relation, relation_label, config):

        obj_proba = np.zeros(len(self.categories_objects))
        room_proba = np.zeros(len(self.categories_rooms))

        if config['context_map']['configure'] == 'grid': # for experiment # 3 -> Occupancy Grid Map
            return obj_proba, room_proba


        self.relation_labels = relation_label

        semantic_spatial_context, agent_location_context = self.context_prompt_generation(semantic_context_map,
                                                                                          spatial_relation)
        room_object, room_room = self.parse_rooms_improved(semantic_spatial_context)

        if self.llm_default == 'Llama2':
            ####### Llama2 ########
            import sys

            first_prompt = self.instruction_configuration_for_Llama2(1, goal_object, room_object, room_room,
                                                                     agent_location_context, config)

            print(f'\n #### First Prompt #### \n {first_prompt}')

            llama2_response = self.chat_llama2(first_prompt)

            first_response = self.room_parse_llm_response_for_Llama2(llama2_response)

            print(f'\n ### First Response ### \n {first_response}')
        else:
            try:
                first_prompt = self.instruction_configuration_for_VertexAI(1, goal_object, room_object, room_room,
                                                                           agent_location_context)
                prompts = [first_prompt]
                gemini_response = self.chat_gemini(prompts)
                first_response = self.room_parse_llm_response_for_VertexAI(gemini_response)

            except:
                plam2_response = self.chat_plam2(prompts)
                first_response = self.room_parse_llm_response_for_VertexAI(plam2_response)

        for room_name, probability in first_response.items():
            room_name = room_name.split('_')[0]
            if room_name in self.categories_rooms:
                index = self.categories_rooms.index(room_name)
                room_proba[index] = probability

        top_two_rooms = sorted(first_response.items(), key=lambda x: x[1], reverse=True)[:2]

        if config['prompting']['type'] == 'chain':
            try:
                if len(top_two_rooms) == 1:
                    selected_instance_items = ','.join(room_object[top_two_rooms[0][0]]['items'])
                elif len(top_two_rooms) == 2:
                    selected_instance_items = ','.join(room_object[top_two_rooms[0][0]]['items'])
                    selected_instance_items += ','.join(room_object[top_two_rooms[1][0]]['items'])
                else:
                    selected_instance_items = "bed_0,book_0,bottle_0,box_0,knife_0,candle_0,chair_0,desk_0,table_0,drawer_0," \
                                              "lamp_0,pencil_0,pot_0,tv stand_0,sofa_0"
            except:
                selected_instance_items = "bed_0,,chair_0,desk_0,table_0,drawer_0," \
                                          "lamp_0,tv stand_0,sofa_0"

        else: # prompting type is "single"
            selected_instance_items = "_0,".join(self.categories_objects)+"_0"



        items = selected_instance_items.split(',')
        # semantic_items = set([item.split('_')[0] for item in items if item])

        semantic_items = set()
        for item in items:
            if item == 'nothing':
                continue
            if item:
                semantic_items.add(item.split('_')[0])

        if self.llm_default == 'Llama2':
            ####### Llama2 ########
            # for i in range(50):
            second_prompt = self.instruction_configuration_for_Llama2(2, goal_object, room_object, room_room,
                                                                      agent_location_context, config,
                                                                      selected_objects=semantic_items)
            print(f'\n #### Second Prompt #### \n {second_prompt}')
            llama2_response = self.chat_llama2(second_prompt)
            second_response = self.object_parse_llm_response_for_Llama2(llama2_response)
            print(f'\n ### Second Response ### \n {second_response}')
            # sys.exit()
        else:
            second_prompt = self.instruction_configuration_for_VertexAI(2, goal_object, room_object, room_room,
                                                                        agent_location_context,
                                                                        selected_objects=semantic_items)
            prompts = [second_prompt]
            try:
                gemini_response = self.chat_gemini(prompts)
                second_response = self.object_parse_llm_response_for_VertexAI(gemini_response)

            except:
                plam2_response = self.chat_plam2(prompts)
                second_response = self.object_parse_llm_response_for_VertexAI(plam2_response)

        for object_name, probability in second_response.items():
            if object_name in self.categories_objects:
                if probability:
                    index = self.categories_objects.index(object_name)
                    obj_proba[index] = probability
        return obj_proba, room_proba

    def context_prompt_generation(self, semantic_context_map, spatial_relation):
        semantic_spatial_context = list(" ")
        agent_location_context = list(" ")

        if len(spatial_relation['inclusion']):
            room_indexs = np.unique(spatial_relation['inclusion'][:, 2])
            # [1] Semantic Spatial Context
            for room_idx in room_indexs:

                contain_sentence = ""
                direction_sentence = ""
                near_sentence = ""

                room_info = semantic_context_map[room_idx]
                room_id = room_info['id']  # ex) badroom_0, 1 ..,2
                if not len(spatial_relation['inclusion']):
                    continue
                indices = np.argwhere(spatial_relation['inclusion'][:, 2] == room_idx)

                for c_idx, idx in enumerate(indices):
                    if c_idx == 0:
                        if len(indices) == 1:
                            contain_sentence += "nothing"
                    else:
                        in_triple = spatial_relation['inclusion'][idx][0]
                        in_subjectIdx = in_triple[0]
                        in_subjectLabel = semantic_context_map[in_subjectIdx]['id']
                        contain_sentence += in_subjectLabel + " "

                        # 방 안에 물체들마다 direction, proximity_near 관계 추출 후 문장 생성

                        # direction relationship
                        if len(spatial_relation['direction']):
                            direction_subIdx = np.argwhere(spatial_relation['direction'][:, 0] == in_subjectIdx)

                            for direction_idx in direction_subIdx:
                                direction_triple = spatial_relation['direction'][direction_idx][0]
                                direction_objLabel = semantic_context_map[direction_triple[2]]['id']
                                direction_relLabel = self.relation_labels[direction_triple[1]]
                                direction_sentence += f'The {in_subjectLabel} is to the {direction_relLabel} {direction_objLabel}, '

                        # proximity_near relationship
                        if len(spatial_relation['proximity_near']):
                            near_subIdx = np.argwhere(spatial_relation['proximity_near'][:, 0] == in_subjectIdx)
                            for near_idx in near_subIdx:
                                near_triple = spatial_relation['proximity_near'][near_idx][0]
                                near_objLabel = semantic_context_map[near_triple[2]]['id']
                                near_relLabel = self.relation_labels[near_triple[1]]
                                near_sentence += f'The {in_subjectLabel} is to the {near_relLabel} {near_objLabel}, '

                sentence = f"{room_id} : contains {contain_sentence}. {direction_sentence}. {near_sentence}. "

                semantic_spatial_context.append(sentence)

        # Room-Room Connection Relationship
        nextTo_sentence = ""

        if len(spatial_relation['proximity_nextTo']):
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
        agent_sentence = ""
        if len(spatial_relation['agent_relation']):
            for agent_triple in spatial_relation['agent_relation']:
                agentIdx = agent_triple[0]
                agentLabel = semantic_context_map[agentIdx]['id']
                relLabel = self.relation_labels[agent_triple[1]]
                objectIdx = agent_triple[2]
                objectLabel = semantic_context_map[objectIdx]['id']
                agent_sentence += f'The {agentLabel} is {relLabel} the {objectLabel}. '
            agent_location_context.append(agent_sentence)

        return semantic_spatial_context, agent_location_context

    def test_chat(self):

        user_input = '''Semantic Spatial Context 
        : bedroom_0 : contains bed_0 book_1 candle_0 chair_2 desk_0 table_2 dresser_1 lamp_0 painting_0 pillow_0 television_0 vase_2 . The bed_0 is to the left of chair_2, The bed_0 is to the left of desk_0, The bed_0 is to the left of table_2, The bed_0 is to the left of dresser_1, The chair_2 is to the right of desk_0, The chair_2 is to the front table_2, The chair_2 is to the behind dresser_1, The desk_0 is to the left of table_2, The desk_0 is to the left of dresser_1, The table_2 is to the behind dresser_1, . The book_1 is to the near table_2, The candle_0 is to the near chair_2, The table_2 is to the near vase_2, . .

        living room_0 : contains bed_2 book_0 bottle_1 cellphone_0 chair_0 desk_1 table_0 drawer_0 dresser_0 lamp_1 newspaper_0 painting_1 pencil_0 pillow_1 plate_1 pot_0 sofa_0 statue_0 apple_0 baseball bat_0 bowl_0 garbage can_0 plant_0 laptop_1 mug_1 vase_1 . The bed_2 is to the left of chair_0, The bed_2 is to the left of desk_1, The bed_2 is to the left of table_0, The bed_2 is to the left of drawer_0, The bed_2 is to the left of dresser_0, The bed_2 is to the left of sofa_0, The chair_0 is to the left of desk_1, The chair_0 is to the left of table_0, The chair_0 is to the left of drawer_0, The chair_0 is to the left of dresser_0, The chair_0 is to the left of sofa_0, The desk_1 is to the front table_0, The desk_1 is to the front drawer_0, The desk_1 is to the front dresser_0, The desk_1 is to the left of sofa_0, The table_0 is to the front drawer_0, The table_0 is to the front dresser_0, The table_0 is to the left of sofa_0, The drawer_0 is to the left of sofa_0, The dresser_0 is to the left of sofa_0, . The cellphone_0 is to the near desk_1, The cellphone_0 is to the near table_0, The desk_1 is to the near newspaper_0, The desk_1 is to the near pencil_0, The desk_1 is to the near plate_1, The desk_1 is to the near statue_0, The desk_1 is to the near bowl_0, The table_0 is to the near newspaper_0, The table_0 is to the near pencil_0, The table_0 is to the near plate_1, The table_0 is to the near statue_0, The table_0 is to the near bowl_0, The drawer_0 is to the near pot_0, The drawer_0 is to the near plant_0, The dresser_0 is to the near pot_0, The dresser_0 is to the near plant_0, The pillow_1 is to the near sofa_0, The sofa_0 is to the near baseball bat_0, The sofa_0 is to the near garbage can_0, . 

        office room_0 : contains bed_1 book_2 bottle_0 chair_1 table_1 lamp_2 fork_0 painting_2 plate_0 laptop_0 mug_0 vase_0 . The bed_1 is to the left of chair_1, The bed_1 is to the left of table_1, The chair_1 is to the right of table_1, . The bed_1 is to the near lamp_2, The bottle_0 is to the near chair_1, The bottle_0 is to the near table_1, The chair_1 is to the near laptop_0, The chair_1 is to the near mug_0, The table_1 is to the near laptop_0, The table_1 is to the near mug_0, . 

        lounge_0 : contains pillow_2

        'The bedroom_0 is next to the living room_0. The bedroom_0 is next to the office room_0. The living room_0 is next to the office room_0. '

        Agent Location Context
        : The robot_agent is positioned at the living room_0. The robot_agent is left of the book_0. The robot_agent is left of the cellphone_0. The robot_agent is left of the desk_1. The robot_agent is left of the table_0. The robot_agent is left of the pencil_0. The robot_agent is left of the statue_0. 

        Given the Semantic Spatial Context  and the Agent Location Context, please estimate the probability of finding an 'alarm clock on a dresser' in each of [bedroom_0, living room_0, office room_0, lounge_0 ]?
        '''

        first_prompt = """
        Semantic Spatial Context:
        - Bedroom_0 contains bed_0, book_1, candle_0, chair_2, desk_0, table_2, dresser_1, lamp_0, painting_0, pillow_0, television_0, and vase_2. Spatial relationships include bed_0 left of chair_2, desk_0, table_2, and dresser_1; chair_2 right of desk_0 and front of table_2; etc.
        - Living Room_0 contains bed_2, book_0, bottle_1, cellphone_0, chair_0, desk_1, table_0, and more. Spatial relationships include bed_2 left of chair_0, desk_1, table_0; chair_0 left of desk_1 and table_0; etc.
        - Office Room_0 contains bed_1, book_2, bottle_0, chair_1, table_1, and more. Spatial relationships include bed_1 left of chair_1 and table_1; chair_1 right of table_1; etc.
        - Lounge_0 contains pillow_2.
        - Proximities: bedroom_0 is next to living room_0 and office room_0; living room_0 is next to office room_0.

        Agent Location Context:
        - The robot_agent is positioned in living room_0, to the left of objects including book_0, cellphone_0, desk_1, table_0, pencil_0, and statue_0.

        Question:
        Given the detailed Semantic Spatial Context of each room, along with the Agent Location Context, please estimate the likelihood of finding an 'alarm clock on a dresser' in each room. 
        Consider general patterns of where an alarm clock and a dresser are usually located in a household, and make an inference based on the provided spatial layout and contents of each room. 
        Provide your estimation as a probability for each of these rooms: bedroom_0, living room_0, office room_0, and lounge_0. Skip the description.
        """
        # Format your response as follows: 'bedroom : [probability],
        # """

        # for i in range(60):
        #     print(f'================{i}===================')
        prompts = [first_prompt]
        self.chat_plam2(prompts)
        self.chat_gemini(prompts)
        second_prompt = """
        Semantic Spatial Context:
        - Bedroom_0 contains bed_0, book_1, candle_0, chair_2, desk_0, table_2, dresser_1, lamp_0, painting_0, pillow_0, television_0, and vase_2. Spatial relationships include bed_0 left of chair_2, desk_0, table_2, and dresser_1; chair_2 right of desk_0 and front of table_2; etc.
        - Living Room_0 contains bed_2, book_0, bottle_1, cellphone_0, chair_0, desk_1, table_0, and more. Spatial relationships include bed_2 left of chair_0, desk_1, table_0; chair_0 left of desk_1 and table_0; etc.
        - Office Room_0 contains bed_1, book_2, bottle_0, chair_1, table_1, and more. Spatial relationships include bed_1 left of chair_1 and table_1; chair_1 right of table_1; etc.
        - Lounge_0 contains pillow_2.
        - Proximities: bedroom_0 is next to living room_0 and office room_0; living room_0 is next to office room_0.

        Agent Location Context:
        - The robot_agent is positioned in living room_0, to the left of objects including book_0, cellphone_0, desk_1, table_0, pencil_0, and statue_0.

        Question:
        Consider general patterns of where an alarm clock and a dresser are usually located in a household, and make an inference based on the provided spatial layout and contents of each room. 
        Based on the Semantic Spatial Context and Agent Location Context, can you provide probability estimations for the likelihood of finding an 'alarm clock on a dresser' in relation to the following objects in bedroom_0: bed_0, book_1, candle_0, chair_2, desk_0, table_2, dresser_1, lamp_0, painting_0, pillow_0, television_0, and vase_2? Please format your response as follows: 'Object Name: Probability Percentage', for each item in the list. For example, 'bed_0: X%, book_1: Y%', and so on, based on your inference from the provided contexts.
        """

        prompts = [second_prompt]
        self.chat_plam2(prompts)
        self.chat_gemini(prompts)
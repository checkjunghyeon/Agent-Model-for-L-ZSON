import argparse
import os
import re
import sys

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import torch

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import str_to_torch_dtype, get_context_length, is_sentence_complete, is_partial_stop
from typing import Optional, Dict, List
from fastchat.Prompt import PromptGenerator

from collections import defaultdict
import json

def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


class Configuration:
    def __init__(self):
        self.awq_ckpt = None
        self.awq_groupsize = -1
        self.awq_wbits = 16
        self.conv_system_msg = None
        self.conv_template = None
        self.cpu_offloading = False
        self.debug = False
        self.device = 'cuda:0'
        self.dtype = None
        self.gptq_act_order = False
        self.gptq_ckpt = None
        self.gptq_groupsize = -1
        self.gptq_wbits = 16
        self.gpus = None
        self.judge_sent_end = False
        self.load_8bit = True
        self.max_gpu_memory = '8GiB'
        self.max_new_tokens = 256
        self.model_path = 'lmsys/vicuna-7b-v1.5'
        self.mouse = False
        self.multiline = False
        self.no_history = False
        self.num_gpus = 1
        self.repetition_penalty = 1.0
        self.revision = 'main'
        self.style = 'simple'
        self.temperature = 1.0



class FastLlama2():
    def __init__(self):
        # args = parser.parse_args()
        args = Configuration()

        args.device = 'cuda:0'
        if args.gpus:
            if len(args.gpus.split(",")) < args.num_gpus:
                raise ValueError(
                    f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        self.model, self.tokenizer, self.gen_params, self.device, self.context_len = self.load_LLM(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            str_to_torch_dtype(args.dtype),
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.conv_system_msg,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            gptq_config=GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ),
            awq_config=AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            ),
            revision=args.revision,
            judge_sent_end=args.judge_sent_end,
            debug=args.debug,
            history=not args.no_history,
        )

    def load_LLM(
            self,
            model_path: str,
            device: str,
            num_gpus: int,
            max_gpu_memory: str,
            dtype: Optional[torch.dtype],
            load_8bit: bool,
            cpu_offloading: bool,
            conv_template: Optional[str],
            conv_system_msg: Optional[str],
            temperature: float,
            repetition_penalty: float,
            max_new_tokens: int,
            gptq_config: Optional[GptqConfig] = None,
            awq_config: Optional[AWQConfig] = None,
            revision: str = "main",
            judge_sent_end: bool = True,
            debug: bool = True,
            history: bool = True,
    ):
        # Model
        model, tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            revision=revision,
            debug=debug,
        )

        context_len = get_context_length(model.config)

        model_type = str(type(model)).lower()
        is_t5 = "t5" in model_type
        is_codet5p = "codet5p" in model_type

        # Hardcode T5's default repetition penalty to be 1.2
        if is_t5 and repetition_penalty == 1.0:
            repetition_penalty = 1.2

        gen_params = {
            "model": model_path,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": None,
            "stop_token_ids": None,
            "echo": False,
        }

        return model, tokenizer, gen_params, device, context_len
        #
        # result = self.generate_stream(model,tokenizer,gen_params,device,context_len)
        # return result

    @torch.inference_mode()
    def generate_stream(
            self,
            prompt,
            stream_interval: int = 2,
            judge_sent_end: bool = False,
    ):

        model = self.model
        tokenizer = self.tokenizer
        params = self.gen_params
        device = self.device
        context_len = self.context_len

        if hasattr(model, "device"):
            device = model.device

        # Read parameters

        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 0.5))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        echo = bool(params.get("echo", False))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(tokenizer.eos_token_id)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )
        input_ids = tokenizer(prompt).input_ids

        if model.config.is_encoder_decoder:
            max_src_len = context_len
        else:  # truncate
            max_src_len = context_len - max_new_tokens - 1

        input_ids = input_ids[-max_src_len:]
        output_ids = list(input_ids)
        input_echo_len = len(input_ids)

        if model.config.is_encoder_decoder:
            encoder_output = model.encoder(
                input_ids=torch.as_tensor([input_ids], device=device)
            )[0]
            start_ids = torch.as_tensor(
                [[model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=device,
            )

        past_key_values = out = None
        sent_interrupt = False
        finish_reason = None
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = model.lm_head(out[0])
                else:
                    out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values
            else:  # decoding
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=torch.as_tensor(
                            [[token] if not sent_interrupt else output_ids],
                            device=device,
                        ),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values if not sent_interrupt else None,
                    )
                    sent_interrupt = False

                    logits = model.lm_head(out[0])
                else:
                    out = model(
                        input_ids=torch.as_tensor(
                            [[token] if not sent_interrupt else output_ids],
                            device=device,
                        ),
                        use_cache=True,
                        past_key_values=past_key_values if not sent_interrupt else None,
                    )
                    sent_interrupt = False
                    logits = out.logits
                past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                _, indices = torch.topk(last_token_logits, 2)
                tokens = [int(index) for index in indices.tolist()]
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                indices = torch.multinomial(probs, num_samples=2)
                tokens = [int(token) for token in indices.tolist()]
            token = tokens[0]
            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            # Yield the output tokens
            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
                if judge_sent_end and stopped and not is_sentence_complete(output):
                    if len(tokens) > 1:
                        token = tokens[1]
                        output_ids[-1] = token
                    else:
                        output_ids.pop()
                    stopped = False
                    sent_interrupt = True

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = is_partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = is_partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")
            if stopped:
                break

        # Finish stream event, which contains finish reason

        output = tokenizer.decode(
            output_ids[input_echo_len:],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        # # Clean
        # del past_key_values
        # gc.collect()
        # torch.cuda.empty_cache()
        # if device == "xpu":
        #     torch.xpu.empty_cache()
        # if device == "npu":
        #     torch.npu.empty_cache()

        return {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }


def read_txt(file):
    with open(file, 'r') as f:
        classes = [line.strip() for line in f]
    return classes


def read_json(file):
    with open(file, 'r') as f:
        rel_json = json.load(f)
    return rel_json


def L3DSG_run_process(mode, classes_list, relation_list, scene_list, LLM, PromptGene):
    if mode == 'obj_knowledge':
        obj_knowledge_dict = dict()
        count = 0
        for scene_idx, scene in enumerate(scene_list):
            temp_dict = defaultdict(list)
            for obj_idx, obj in enumerate(classes_list):

                prompt = PromptGene.object_prompt_generation(obj, scene)
                output = LLM.generate_stream(prompt)

                subkey = obj
                temp_dict[subkey].append(output['text'])
                count += 1

                if count % 10 == 0:
                    print(output['text'])
                    print('scene : {}/{} \n object : {}/{}'.format(
                        scene_idx, len(scene_list) + 1,
                        obj_idx, len(classes_list) + 1
                    ))

            obj_knowledge_dict[scene] = dict(temp_dict)
            with open('knowledge_db/obj_knowledge_in_{}.json'.format(scene), 'w') as obj_file:
                json.dump(obj_knowledge_dict, obj_file)
                print('save {} db'.format(scene))
                obj_knowledge_dict = dict()

    elif mode == 'rel_knowledge':
        rel_knowledge_dict = dict()
        count = 0
        for sceneIdx, scene in enumerate(scene_list):
            rel_dict = defaultdict(list)
            for subIdx, subject in enumerate(classes_list):
                for objIdx, object in enumerate(classes_list):
                    for relIdx, relation in enumerate(relation_list):
                        pred_mode = 'SPO'
                        prompt = PromptGene.relation_prompt_generation(pred_mode, subject, relation,
                                                                       object, scene)
                        output = LLM.generate_stream(prompt)
                        subkey = "{}-{}-{}".format(subject, relation, object)
                        rel_dict[subkey].append(output['text'])
                        count += 1

                        if count % 20 == 0:
                            print(output['text'])
                            print('scene : {}/{} \n subject : {}/{} \n object : {}/{} \n predicate : {}/{}'.format(
                                sceneIdx, len(scene_list) + 1,
                                subIdx, len(classes_list) + 1,
                                objIdx, len(classes_list) + 1,
                                relIdx, len(relation_list) + 1,

                            ))

            rel_knowledge_dict[scene] = dict(rel_dict)
            with open('knowledge_db/rel_knowledge_in_{}.json'.format(scene), 'w') as rel_file:
                json.dump(rel_knowledge_dict, rel_file)
                print('save {} db'.format(scene))
                rel_knowledge_dict = dict()

    elif mode == 'gt_label':
        train_rel_json = read_json('3DSSG/relationships_train.json')['scans']
        val_rel_json = read_json('3DSSG/relationships_validation.json')['scans']

        for sceneIdx, scene in enumerate(scene_list):
            for train_scene in train_rel_json:
                objects_dict = train_scene['objects']
                relations_list = train_scene['relationships']
                for relation in relations_list:
                    subject = objects_dict[str(relation[0])]
                    object = objects_dict[str(relation[1])]
                    predicate = relations[-1]


    elif mode == 'room_type':

        train_scans = read_txt('3DSSG/train_scans.txt')
        val_scans = read_txt('3DSSG/validation_scans.txt')
        train_rel_json = read_json('3DSSG/relationships_train.json')['scans']
        val_rel_json = read_json('3DSSG/relationships_validation.json')['scans']
        origin_room_type = read_json('3DSSG/room_type.json')
        temp_dict = {}

        total_scan_dict = {}
        for train_scan in train_rel_json:
            scan_id = train_scan['scan']
            total_scan_dict[scan_id] = train_scan
        for val_scan in val_rel_json:
            scan_id = val_scan['scan']
            total_scan_dict[scan_id] = val_scan
        new_dict = {}
        for scan_id, room_type in origin_room_type.items():
            if room_type == "room":
                select_scan = total_scan_dict[scan_id]
                prompt = PromptGene.room_type_prompt(select_scan['objects'])
                output = LLM.generate_stream(prompt)['text']
                output = output.lower()
                print('Output : ', output)
                new_dict[scan_id] = output
            else:
                new_dict[scan_id] = room_type
        with open('3DSSG/0922_room_type.json', 'w') as room_file:
            json.dump(new_dict, room_file)


def LZSON_run_process(LLM, PromptGene):
    user_input = '''Robot Agent Centric Context
: The robot is located adjacent to the bedroom. It is positioned far from the bed, television, pillow, and lamp.

Area-specific Contexts
Area1 : There is a bedroom with a bed, television, pillow, lamp.The bed is to the right of the television.The bed is to the left of the pillow.The bed is near the lamp.The bed is to the left of the lamp.The television is to the left of the pillow.The television is to the left of the lamp.The pillow is to the right of the lamp.

Area2 : There are no objects in the office room.

Area3 : There is a living room with a book, bowl, plant, drawer, painting, pencil, pot, sofa, statue.The book is to the right of the bowl.The book is southeast of the plant.The book is to the left of the drawer.The book is southeast of the painting.The book is near the pencil.The book is to the right of the pencil.The book is to the left of the pot.The book is to the right of the sofa.The book is near the statue.The book is to the right of the statue.The bowl is to the left of the plant.The bowl is to the left of the drawer.The bowl is southwest of the painting.The bowl is near the pencil.The bowl is to the left of the pencil.The bowl is to the left of the pot.The bowl is to the right of the sofa.The bowl is near the statue.The bowl is to the left of the statue.The plant is near the drawer.The plant is to the left of the drawer.The plant is near the painting.The plant is to the right of the painting.The plant is to the right of the pencil.The plant is near the pot.The plant is to the left of the pot.The plant is to the right of the sofa.The plant is northeast of the statue.The drawer is to the right of the painting.The drawer is to the right of the pencil.The drawer is near the pot.The drawer is to the left of the pot.The drawer is southeast of the sofa.The drawer is to the right of the statue.The painting is northeast of the pencil.The painting is to the left of the pot.The painting is to the right of the sofa.The painting is northwest of the statue.The pencil is to the left of the pot.The pencil is to the right of the sofa.The pencil is near the statue.The pencil is to the left of the statue.The pot is northeast of the sofa.The pot is to the right of the statue.The sofa is to the left of the statue.

Given the robot agent-centric context and the area-specific context, please estimate the probability of finding an 'alarm clock on a dresser' in each of these areas.
Format your response as follows: 'Area1: [probability], Area2: [probability], Area3: [probability]'. Use a scale from 0 to 1 for the probabilities.'''

    user_input = '''Answer the question based on the robot agent-centric context and area-specific context below. 
    Please estimate the probability of finding an 'alarm clock on a dresser' in each of these areas. Provide a probability score between 0 and 1.

    Robot Agent Centric Context
    : The robot is located adjacent to the bedroom. It is positioned far from the bed, television, pillow, and lamp.

    Area-specific Contexts
    -Area1 : There is a bedroom with a bed, television, pillow, lamp.The bed is to the right of the television.The bed is to the left of the pillow.The bed is near the lamp.The bed is to the left of the lamp.The television is to the left of the pillow.The television is to the left of the lamp.The pillow is to the right of the lamp.

    -Area2 : There are no objects in the office room.

    -Area3 : There is a living room with a book, bowl, plant, drawer, painting, pencil, pot, sofa, statue.The book is to the right of the bowl.The book is southeast of the plant.The book is to the left of the drawer.The book is southeast of the painting.The book is near the pencil.The book is to the right of the pencil.The book is to the left of the pot.The book is to the right of the sofa.The book is near the statue.The book is to the right of the statue.The bowl is to the left of the plant.The bowl is to the left of the drawer.The bowl is southwest of the painting.The bowl is near the pencil.The bowl is to the left of the pencil.The bowl is to the left of the pot.The bowl is to the right of the sofa.The bowl is near the statue.The bowl is to the left of the statue.The plant is near the drawer.The plant is to the left of the drawer.The plant is near the painting.The plant is to the right of the painting.The plant is to the right of the pencil.The plant is near the pot.The plant is to the left of the pot.The plant is to the right of the sofa.The plant is northeast of the statue.The drawer is to the right of the painting.The drawer is to the right of the pencil.The drawer is near the pot.The drawer is to the left of the pot.The drawer is southeast of the sofa.The drawer is to the right of the statue.The painting is northeast of the pencil.The painting is to the left of the pot.The painting is to the right of the sofa.The painting is northwest of the statue.The pencil is to the left of the pot.The pencil is to the right of the sofa.The pencil is near the statue.The pencil is to the left of the statue.The pot is northeast of the sofa.The pot is to the right of the statue.The sofa is to the left of the statue.


    Question: Given the robot agent-centric context and the area-specific context, please estimate the probability of finding an 'alarm clock on a dresser' in each of these areas in [0,1].
     Format your response as follows: 'Frontier Area k: [probability],'.\n'''

    user_input_0307 = '''
    Semantic Spatial Context 
: bedroom_0 : contains bed_0 book_1 candle_0 chair_2 desk_0 table_2 dresser_1 lamp_0 painting_0 pillow_0 television_0 vase_2 . The bed_0 is to the left of chair_2, The bed_0 is to the left of desk_0, The bed_0 is to the left of table_2, The bed_0 is to the left of dresser_1, The chair_2 is to the right of desk_0, The chair_2 is to the front table_2, The chair_2 is to the behind dresser_1, The desk_0 is to the left of table_2, The desk_0 is to the left of dresser_1, The table_2 is to the behind dresser_1, . The book_1 is to the near table_2, The candle_0 is to the near chair_2, The table_2 is to the near vase_2, . .

living room_0 : contains bed_2 book_0 bottle_1 cellphone_0 chair_0 desk_1 table_0 drawer_0 dresser_0 lamp_1 newspaper_0 painting_1 pencil_0 pillow_1 plate_1 pot_0 sofa_0 statue_0 apple_0 baseball bat_0 bowl_0 garbage can_0 plant_0 laptop_1 mug_1 vase_1 . The bed_2 is to the left of chair_0, The bed_2 is to the left of desk_1, The bed_2 is to the left of table_0, The bed_2 is to the left of drawer_0, The bed_2 is to the left of dresser_0, The bed_2 is to the left of sofa_0, The chair_0 is to the left of desk_1, The chair_0 is to the left of table_0, The chair_0 is to the left of drawer_0, The chair_0 is to the left of dresser_0, The chair_0 is to the left of sofa_0, The desk_1 is to the front table_0, The desk_1 is to the front drawer_0, The desk_1 is to the front dresser_0, The desk_1 is to the left of sofa_0, The table_0 is to the front drawer_0, The table_0 is to the front dresser_0, The table_0 is to the left of sofa_0, The drawer_0 is to the left of sofa_0, The dresser_0 is to the left of sofa_0, . The cellphone_0 is to the near desk_1, The cellphone_0 is to the near table_0, The desk_1 is to the near newspaper_0, The desk_1 is to the near pencil_0, The desk_1 is to the near plate_1, The desk_1 is to the near statue_0, The desk_1 is to the near bowl_0, The table_0 is to the near newspaper_0, The table_0 is to the near pencil_0, The table_0 is to the near plate_1, The table_0 is to the near statue_0, The table_0 is to the near bowl_0, The drawer_0 is to the near pot_0, The drawer_0 is to the near plant_0, The dresser_0 is to the near pot_0, The dresser_0 is to the near plant_0, The pillow_1 is to the near sofa_0, The sofa_0 is to the near baseball bat_0, The sofa_0 is to the near garbage can_0, . 

office room_0 : contains bed_1 book_2 bottle_0 chair_1 table_1 lamp_2 fork_0 painting_2 plate_0 laptop_0 mug_0 vase_0 . The bed_1 is to the left of chair_1, The bed_1 is to the left of table_1, The chair_1 is to the right of table_1, . The bed_1 is to the near lamp_2, The bottle_0 is to the near chair_1, The bottle_0 is to the near table_1, The chair_1 is to the near laptop_0, The chair_1 is to the near mug_0, The table_1 is to the near laptop_0, The table_1 is to the near mug_0, . 

lounge_0 : contains pillow_2

'The bedroom_0 is next to the living room_0. The bedroom_0 is next to the office room_0. The living room_0 is next to the office room_0. '

Agent Location Context
: The robot_agent is positioned at the living room_0. The robot_agent is left of the book_0. The robot_agent is left of the cellphone_0. The robot_agent is left of the desk_1. The robot_agent is left of the table_0. The robot_agent is left of the pencil_0. The robot_agent is left of the statue_0. 

Given the Semantic Spatial Context  and the Agent Location Context, please estimate the probability of finding an 'alarm clock on a dresser' in each of [bedroom_0, living room_0, office room_0, lounge_0 ]?
Format your response as follows: 'bedroom : [probability],'.\n
'''

    user_input_new = """
    Semantic Spatial Context:
    - Bedroom_0 contains bed_0, book_1, candle_0, chair_2, desk_0, table_2, dresser_1, lamp_0, painting_0, pillow_0, television_0, and vase_2. Spatial relationships include bed_0 left of chair_2, desk_0, table_2, and dresser_1; chair_2 right of desk_0 and front of table_2; etc.
    - Living Room_0 contains bed_2, book_0, bottle_1, cellphone_0, chair_0, desk_1, table_0, and more. Spatial relationships include bed_2 left of chair_0, desk_1, table_0; chair_0 left of desk_1 and table_0; etc.
    - Office Room_0 contains bed_1, book_2, bottle_0, chair_1, table_1, and more. Spatial relationships include bed_1 left of chair_1 and table_1; chair_1 right of table_1; etc.
    - Lounge_0 contains pillow_2.
    - Proximities: bedroom_0 is next to living room_0 and office room_0; living room_0 is next to office room_0.

    Agent Location Context:
    - The robot_agent is positioned in living room_0, to the left of objects including book_0, cellphone_0, desk_1, table_0, pencil_0, and statue_0.

    Question:
    Considering the Semantic Spatial Context and the Agent Location Context, can you estimate the probability of finding an 'alarm clock on a dresser' in each of these rooms: bedroom_0, living room_0, office room_0, and lounge_0?
    Please provide the probability as a numeric value for each room..\n
    """
    prompt = PromptGene.user_prompt_generation(user_input_new)
    output = LLM.generate_stream(prompt)
    print(output)


if __name__ == "__main__":
    # object_list = read_txt('3DSSG/classes.txt')
    # scene_type = ["living room","toilet","dining room","bed room","kitchen room","storage room","desk room","room"]
    # relation_list = read_txt('3DSSG/relations.txt')
    # scene_type = ["dining room","bed room","kitchen room","storage room","desk room","living room", "toilet"]

    localLLM = FastLlama2()
    PromptGene = PromptGenerator()
    # mode = 'obj_knowledge'
    # run_process(mode, object_list, relation_list, scene_type, localLLM, PromptGene)
    # mode = 'rel_knowledge'
    # run_process(mode, object_list, relation_list, scene_type, localLLM, PromptGene)

    # mode = 'room_type'
    # L3DSG_run_process(mode, object_list, relation_list, scene_type, localLLM, PromptGene)

    LZSON_run_process(localLLM, PromptGene)


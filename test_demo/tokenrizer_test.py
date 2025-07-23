from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
import trl
from trl.data_utils import maybe_apply_chat_template
import torch

model_init_kwargs = {}

processing_class = AutoProcessor.from_pretrained("/LLM-VLM/train_saves/Qwen2.5-VL-7b-424-240-9-usample-r2r/merged-950", trust_remote_code=model_init_kwargs.get("trust_remote_code", None))

# example ={
#         'goal_position': [-3.2167999744415283, 0.07484699785709381, -0.9348180294036865], 
#         'distance_to_goal': 3.2942721843719482, 
#         'agent_position': [-6.409084796905518, -0.12146295607089996], 
#         'agent_heading': -1.5707963705062866, 
#         'image_path': ['/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_0.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_3.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_9.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_15.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_17.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_21.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_26.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_30.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_32.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_35.jpg'], 
#         'problem': 'You are a robot programmed for navigation tasks. You have been given a series of historical observations: <image> <image> <image> <image> <image> <image> <image> <image> <image> and the current observation: <image>. Your last action is: "move forward 0.5 meters". Your assigned task is: "Go to the left side of the dining table. Go to the small vase on the floor. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'solution': 'turn left 15 degrees', 
#         'accu_reward_method': 'default', 
#         'prompt': [
#             {
#                 'content': 
#                 [
#                     {'image': None, 'text': 'You are a robot programmed for navigation tasks. You have been given a series of historical observations:', 'type': 'text'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_0.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_3.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_9.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_15.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_17.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_21.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_26.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_30.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_32.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': None, 'text': 'and the current observation:', 'type': 'text'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_35.jpg', 'text': None, 'type': 'image'}, 
#                     {'image': None, 'text': '. Your last action is: "move forward 0.5 meters". Your assigned task is: "Go to the left side of the dining table. Go to the small vase on the floor. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'type': 'text'}
#                 ], 
#                 'role': 'user'
#             }
#         ]
#     }

example_messages = {
    "messages": [
      {
        "content": "You are a robot programmed for navigation tasks. You have been given current observation: <image>. Your assigned task is: \"Walk towards the white bookshelf. Turn left at the white bookshelf. Stop in between the two tables. \". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.",
        "role": "user"
      },
      {
        "content": "turn right 45 degrees",
        "role": "assistant"
      }
    ],
    "images": [
      "data/vln_datasets/r2r_train_424_240/9224/9224_0.jpg"
    ]
}

# example ={
#         'goal_position': [-3.2167999744415283, 0.07484699785709381, -0.9348180294036865], 
#         'distance_to_goal': 3.2942721843719482, 
#         'agent_position': [-6.409084796905518, -0.12146295607089996], 
#         'agent_heading': -1.5707963705062866, 
#         'image_path': ['/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_0.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_3.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_9.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_15.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_17.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_21.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_26.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_30.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_32.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_35.jpg'], 
#         'problem': 'You are a robot programmed for navigation tasks. You have been given a series of historical observations: <image> <image> <image> <image> <image> <image> <image> <image> <image> and the current observation: <image>. Your last action is: "move forward 0.5 meters". Your assigned task is: "Go to the left side of the dining table. Go to the small vase on the floor. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'solution': 'turn left 15 degrees', 
#         'accu_reward_method': 'default', 
#         'prompt': [
#             {
#                 'content': 
#                 [
#                     {'text': 'You are a robot programmed for navigation tasks. You have been given a series of historical observations:', 'type': 'text'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_0.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_3.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_9.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_15.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_17.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_21.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_26.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_30.jpg', 'type': 'image'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_32.jpg', 'type': 'image'}, 
#                     {'text': 'and the current observation:', 'type': 'text'}, 
#                     {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/7369/7369_35.jpg', 'type': 'image'}, 
#                     {'text': '. Your last action is: "move forward 0.5 meters". Your assigned task is: "Go to the left side of the dining table. Go to the small vase on the floor. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'type': 'text'}
#                 ], 
#                 'role': 'user'
#             }
#         ]
#     }

# example ={'goal_position': [6.5514397621154785, 0.09357202053070068, -3.5540199279785156], 
#           'distance_to_goal': 7.879015922546387, 
#           'agent_heading': 1.0471975803375244, 
#           'image_path': ['/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_0.jpg'], 
#           'problem': 'You are a robot programmed for navigation tasks. You have been given the current observation: <image>. Your assigned task is: "Walk towards the white bookshelf. Turn left at the white bookshelf. Stop in between the two tables. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'solution': 'turn right 45 degrees', 
#           'prompt': [
#               {'content': [
#                   {'image': None, 'text': 'You are a robot programmed for navigation tasks. You have been given the current observation:', 'type': 'text'}, 
#                   {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_0.jpg', 'text': None, 'type': 'image'}, 
#                   {'image': None, 'text': '. Your assigned task is: "Walk towards the white bookshelf. Turn left at the white bookshelf. Stop in between the two tables. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'type': 'text'}
#                   ],
#                'role': 'user'}]
#           }

example = {'goal_position': [6.5514397621154785, 0.09357202053070068, -3.5540199279785156], 'distance_to_goal': 5.452889442443848, 'agent_heading': 5.960464477539063e-08, 'image_path': ['/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_0.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_3.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_6.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_8.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_11.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_14.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_15.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_17.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_20.jpg', '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_23.jpg'], 'problem': 'You are a robot programmed for navigation tasks. You have been given a series of historical observations: <image> <image> <image> <image> <image> <image> <image> <image> <image> and the current observation: <image>. Your last action is: "move forward 0.75 meters". Your assigned task is: "Walk towards the white bookshelf. Turn left at the white bookshelf. Stop in between the two tables. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'solution': 'move forward 0.75 meters', 'prompt': [{'content': [{'image': None, 'text': 'You are a robot programmed for navigation tasks. You have been given a series of historical observations:', 'type': 'text'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_0.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_3.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_6.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_8.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_11.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_14.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_15.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_17.jpg', 'text': None, 'type': 'image'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_20.jpg', 'text': None, 'type': 'image'}, {'image': None, 'text': 'and the current observation:', 'type': 'text'}, {'image': '/LLM-VLM/datasets/rl_vln/r2r_train_424_240/9224/9224_23.jpg', 'text': None, 'type': 'image'}, {'image': None, 'text': '. Your last action is: "move forward 0.75 meters". Your assigned task is: "Walk towards the white bookshelf. Turn left at the white bookshelf. Stop in between the two tables. ". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete.', 'type': 'text'}], 'role': 'user'}]}

# prompts_text = maybe_apply_chat_template(example, processing_class)["prompt"]



def clean_prompt(example):
    for message in example["prompt"]:
        cleaned_content = []
        for item in message["content"]:
            if item["type"] == "text":
                cleaned_content.append({
                    "type": "text",
                    "text": item["text"]
                })
            elif item["type"] == "image":
                cleaned_content.append({
                    "type": "image",
                    "image": item["image"]
                })
        message["content"] = cleaned_content
    return example

prompts_text = maybe_apply_chat_template(clean_prompt(example), processing_class)
print(prompts_text)

# print(trl.data_utils.__file__)
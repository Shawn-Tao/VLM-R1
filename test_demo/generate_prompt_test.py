from datasets import Dataset, load_dataset, load_from_disk
import json
import os

data_file_paths="/LLM-VLM/datasets/rl_vln/r2r_rl_train_424_240.jsonl"
image_folders="/LLM-VLM/datasets/rl_vln/r2r_train_424_240"


data_files = data_file_paths.split(":")
image_folders = image_folders.split(":")

if len(data_files) != len(image_folders):
    raise ValueError("Number of data files must match number of image folders")

if len(data_files) != len(image_folders):
    raise ValueError("Number of data files must match number of image folders")

all_data = []
for data_file, image_folder in zip(data_files, image_folders):
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'image' in item:
                if isinstance(item['image'], str):
                    # Store image path instead of loading the image
                    item['image_path'] = [os.path.join(image_folder, item['image'])]
                    del item['image'] # remove the image column so that it can be loaded later
                elif isinstance(item['image'], list):
                    # if the image is a list, then it is a list of images (for multi-image input)
                    item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                    del item['image'] # remove the image column so that it can be loaded later
                else:
                    raise ValueError(f"Unsupported image type: {type(item['image'])}")
            # !Don't Remove immediate image loading
            # item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
            item['problem'] = item['conversations'][0]['value']
            
            # Handle solution that could be a float or string
            solution_value = item['conversations'][1]['value']
            if isinstance(solution_value, str):
                item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
            else:
                # If it's a float or other non-string type, keep it as is
                item['solution'] = str(solution_value)
            
            del item['conversations']
            all_data.append(item)

dataset = Dataset.from_list(all_data)
dataset = dataset.select(range(16))
print(dataset[0])


def make_conversation_from_jsonl(example):
    image_paths = example.get("image_path", [])
    num_images = len(image_paths)
    assert all(os.path.exists(p) for p in image_paths), f"Image paths do not exist: {image_paths}"

    # 将 prompt 按 <image> 分割为文本片段
    prompt_template = example['problem']
    text_parts = prompt_template.split("<image>")

    # 构造 content：每个 text 后插一个 image（image 可能比 text 少一个）
    content = []
    for i, text in enumerate(text_parts):
        if text.strip():
            content.append({'type': 'text', 'text': text.strip()})  # 只含 text
        if i < num_images:
            content.append({'type': 'image', 'image': image_paths[i]})  # 只含 image

    return {
        'image_path': image_paths,
        'problem': prompt_template,
        'solution': example['solution'],
        # 结构化输入方式，QwenVL的autoprocessor会自动进行处理，图像文本拼接后送入网络
        'prompt': [{
            'role': 'user',
            'content': content
        }],
        'goal_position':example['goal_position'],
        'distance_to_goal':example['distance_to_goal'],
        'agent_heading':example['agent_heading'],
    }
    
    
    
dataset = dataset.map(make_conversation_from_jsonl,  remove_columns=dataset.column_names, num_proc=2)

print(dataset[9])
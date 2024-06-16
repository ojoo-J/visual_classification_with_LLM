import os
import openai
import json
import itertools

import torch
import numpy as np
from openai import OpenAI 
from load import *
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from descriptor_strings import stringtolist

def generate_prompt(category_name: str, related_words: list):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo? The caption words related to "lemur" are 'standing', 'tree', 'lemurs', 'sitting', 'birds'.
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo? The caption words related to "television" are 'sitting', 'and', 'bunch', 'room', 'television'.
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful visual features for distinguishing a {category_name} in a photo? The caption words related to "{category_name}" are {related_words[0]}, {related_words[1]}, {related_words[2]}, {related_words[3]}, {related_words[4]}.
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
            
    
def obtain_descriptors(class_list, related_words, gpt_model = "gpt-4o"):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt(category.replace('_', ' '), related_words) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "system", "content": "Please create a continuation of A in the same format as the given prompt."}, # <-- This is the system message that provides context to the model
                  {"role": "user", "content": prompts[0]}  # <-- This is the user message for which the model will generate a response
                  ])
    # response_texts = [r["text"] for resp in responses for r in [completion.choices[0].message.content]]
    # descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors_list = stringtolist(completion.choices[0].message.content)
    if len(descriptors_list) > 10:
        return descriptors_list[:10]
    else:
        return descriptors_list
        


if __name__ == "__main__":
    
    seed = 0
    device = 'cuda:5'
    seed_everything(seed)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", f'{your_key}')) #FILL IN YOUR OWN HERE
    
    orig_json_path = '/data2/youngju/visual_classification_with_LLM/descriptors/baseline/descriptors_imagenet.json'
    with open(orig_json_path, "r") as json_file:
        orig_json_data = json.load(json_file)
    label_list = list(orig_json_data.keys())
    
    gpt_model = "gpt-4o"
    
    save_path = f'/data2/youngju/clip_classifier/classify_by_description_release/descriptors/{gpt_model}'
    os.makedirs(save_path, exist_ok=True)
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.eval()
    blip_model.to(device)
    
    bs = hparams['batch_size']
    dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=4, pin_memory=True)
    
    
    desc_dict = {}
    for target_label in tqdm(range(1000)):
        target_idx = np.where(np.array(dataloader.dataset.targets) == target_label)[0]
        captions = []
        label = label_list[target_label]
        
        # target label에 해당하는 캡션 모으기
        for idx in target_idx:
            x = dataloader.dataset[idx][0].unsqueeze(dim=0).to(device)
            outputs = blip_model.generate(x, num_return_sequences=1, num_beams=2, max_length=20, early_stopping=True) # num_beams=num_captions,
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)
            
        frequency = dict()
        for cap in captions:
            for word in cap.split(' '):
                if word not in ['a', 'of', 'the', 'on', 'in', 'with', 'group', 'row']:
                    if word in frequency:
                        frequency[word] += 1
                    else:
                        frequency[word] = 1
        words = list(frequency.keys())
        freq = list(frequency.values())
        
        min_idx = np.array(freq).argsort()
        related_words = list(np.array(words)[min_idx[-5:]])
        
        desc = obtain_descriptors([label], related_words, gpt_model = gpt_model)
        desc_dict[f'{label}'] = desc
        
    with open(f'{save_path}/descriptors_imagenet_with_related_words.json', 'w') as fp:
        json.dump(desc_dict, fp, indent=4)
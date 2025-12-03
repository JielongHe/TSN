

import json

import dashscope

def generate_text_for_image(image_path):

    try:
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_path},
                    {"text": "Describe the appearance of the pedestrian according to the following template, outputting only the description,"
                                " \"The {gender} is wearing a {upper_color} {upper_garment} with {lower_color} {lower_garment} and {shoe_color}{shoes}. The {gender} has {hairstyle} {hair_color} hair andappears to be {age_group}. The {gender} is carrying {belongings}.\""
                                }
                ]
            }
        ]


        response = dashscope.MultiModalConversation.call(
            api_key='sk-46db4d1acd504df0b65395d70e5896e7',
            model='qwen-vl-max',
            messages=messages
        )


        if response.status_code == 200:
            generated_text = response.output.choices[0].message.content[0]['text']
            return generated_text
        else:
            print(f"API request failed with status: {response.status_code}")
            print(f"Error: {response.message}")
            return None

    except Exception as e:
        print(f"Error generating text for {image_path}: {e}")
        return None


def process_images_in_folder(train_list):
    results = []

    for i, ann in enumerate(train_list):

        img_path = os.path.join('./data/CUHK-PEDES/imgs',ann['img_path'])

        generated_text = generate_text_for_image(img_path)

        result = {
            "id": ann["id"],
            "image_path": img_path,
            "captions": ann["captions"],
            "generated_text": generated_text
        }
        results.append(result)

    with open(f"pre_gen_data/cuhk_qwen_gen_captions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)




import os
if __name__ == "__main__":
    root_dir = './data/CUHK-PEDES'
    raw_dir = 'reid_raw.json'

    with open(os.path.join(root_dir, raw_dir), 'r') as f:
        cap_list = json.load(f)

    train_list = []
    val_list = []
    test_list = []

    for info in cap_list:
        if info['split'] == 'train':
            info1 = info.copy()
            info1['captions'] = info['captions'][0]
            train_list.append(info1)
        elif info['split'] == 'test':
            test_list.append(info)
        else:
            val_list.append(info)

    process_images_in_folder(train_list)
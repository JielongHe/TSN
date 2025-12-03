

import json
from pathlib import Path
import dashscope
import os
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

        # 发送API请求
        response = dashscope.MultiModalConversation.call(
            api_key='sk-46db4d1acd504df0b65395d70e5896e7',
            model='qwen-vl-max',
            messages=messages
        )

        # 检查响应状态并获取生成的文本
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


def process_images_in_folder(test_lists):
    results=[]
    i = 0
    for test_list in test_lists:


        gen_captons = []
        caption1 = test_list["captions"][0]

        prompte1 = f"Generate text based on the template “The {{gender}} is wearing a {{upper_color}} {{upper_garment}} with {{lower_color}}{{lower_garment}} and {{shoe_color}}{{shoes}}. The {{gender}} has {{hairstyle}} {{hair_color}} hair and appears to be {{age_group}}. The {{gender}} is carrying {{belongings}}.” " \
                   f"using the text “{caption1}” Only output the template results and exclude any attributes that are not mentioned in the input (such as hairstyle, hair color, age group, or belongings if they are not specified)."

        messages = [
            {'role': 'user', 'content': prompte1}
        ]

        response = dashscope.Generation.call(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
            api_key='sk-46db4d1acd504df0b65395d70e5896e7',
            model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
            messages=messages,
            # result_format参数不可以设置为"text"。
            result_format='message'
        )

        try:
            gen_caption = response.output.choices[0].message.content
        except:
            gen_caption = "null"

        gen_captons.append(gen_caption)


        image_path = os.path.join('./data/CUHK-PEDES/imgs', test_list['file_path'])

        img_gen_text = generate_text_for_image(image_path)


        result = {
            "split": test_list['split'],
            "id": test_list["id"],
            "image_path": test_list["file_path"],
            "img_gen_text": img_gen_text,
            "captions": test_list["captions"],
            "generated_text": gen_captons
        }
        results.append(result)


    with open(f'pre_gen_data/cuhk_test_qwen_gen_captions.json', "w") as file:
        json.dump(results, file, indent=4)




if __name__ == "__main__":
    root_dir = './data/CUHK-PEDES'
    raw_dir = 'reid_raw.json'

    with open(os.path.join(root_dir, raw_dir), 'r') as f:
        cap_list = json.load(f)

    train_list, val_list, test_list = [], [], []
    for cap in cap_list:
        if cap['split'] == 'train':
            train_list.append(cap)
        elif cap['split'] == 'test':
            test_list.append(cap)
        else:
            val_list.append(cap)

    process_images_in_folder(test_list)

import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption

def split_RSTP_PEDE():
    root_dir = '/home/aorus/He/data/RSTPReid'
    raw_dir = 'data_captions.json'

    with open(os.path.join(root_dir, raw_dir), 'r') as f:
        cap_list = json.load(f)

    train_list = []
    val_list = []
    test_list = []

    for info in cap_list:
        if info['split'] == 'train':
            info1 = info.copy()
            # info2 = info.copy()
            info1['captions'] = info['captions'][0]
            # info2['captions'] = info['captions'][1]
            train_list.append(info1)
            # train_list.append(info2)
        elif info['split'] == 'test':
            test_list.append(info)
        else:
            val_list.append(info)

    return train_list, val_list, test_list

class rstp_pre_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):

        self.train_list, _, _= split_RSTP_PEDE()
        output_file = "./pre_gen_data/rstp_qwen_gen_captions.json"
        with open(output_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        self.annotation = train_data

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.train_list:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        # image_path = self.image_root
        img_path = ann['image_path']
        image = Image.open(img_path)
        image = self.transform(image)
        captions = ann['generated_text']
        # captions = captions.split('.')[0]

        if captions==None:
            print(1)
        captions = self.prompt + pre_caption(captions, self.max_words)

        # multi_captions = ann['gen_caption']
        # multi_captions = self.prompt + pre_caption(multi_captions, self.max_words)
        multi_captions = ''

        gpt_gen_caption = ''

        # gpt_gen_caption = self.prompt + pre_caption(gpt_gen_caption, self.max_words)

        return image, captions, multi_captions, self.img_ids[ann['id']], img_path, gpt_gen_caption


import random
class rstp_caption_gen_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):
        train_list, _, _= split_RSTP_PEDE()

        random_select = random.sample(train_list, 5000)
        self.annotation = random_select
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['img_path']
        img_path = os.path.join(self.image_root, image_path)
        image = Image.open(img_path)
        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, img_path, self.img_ids[ann['id']]


class rstp_caption_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):
        train_list, _, _= split_RSTP_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['img_path']
        img_path = os.path.join(self.image_root, image_path)
        image = Image.open(img_path)
        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, img_path, self.img_ids[ann['id']]

class rstp_pre_train1(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):

        output_file = "./pre_gen_data/rstp_gen_caption.json"
        with open(output_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        self.annotation = train_data
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['image_path']
        image = Image.open(image_path)
        image = self.transform(image)
        captions = ann['gen_caption']
        captions = self.prompt + pre_caption(captions, self.max_words)

        return image, captions

class pre_rstp_pede_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):

        output_file = './pre_gen_data/rstp_gen_pre_caption.json'
        with open(output_file, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        rand_sample = random.sample(combined_data, 4000)
        self.annotation = rand_sample
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root,image_path))
        image = self.transform(image)
        captions = ann['gen_caption']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

class rstp_pede_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):

        train_list, _, _= split_RSTP_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['img_path']
        img_path = os.path.join(self.image_root, image_path)
        image = Image.open(img_path)
        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']], img_path

class rstp_pede_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, split, max_words=72):
        self.max_words = max_words
        output_file = "./pre_gen_data/rstp_test_qwen_gen_captions.json"
        with open(output_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        self.annotation = train_data

        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.gen_text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.txt2pid = []
        self.img2pid = []

        person = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image_path'])

            caps = ann['captions']

            self.text.append(pre_caption(caps[0], max_words))
            self.text.append(pre_caption(caps[1], max_words))

            caps1 = ann['generated_text']

            self.gen_text.append(pre_caption(caps1[0], max_words))
            self.gen_text.append(pre_caption(caps1[1], max_words))

            pid = ann['id']
            self.img2pid.append(pid)
            self.txt2pid.append(pid)
            self.txt2pid.append(pid)
            if pid not in person.keys():
                person[pid] = {'image': [img_id], 'text': [txt_id, txt_id + 1]}
            else:
                person[pid]['image'].append(img_id)
                person[pid]['text'].append(txt_id)
                person[pid]['text'].append(txt_id + 1)
            txt_id = txt_id + 2

        for pid in person.keys():
            for img_id in person[pid]['image']:
                self.img2txt[img_id] = person[pid]['text']
                assert self.img2pid[img_id] == pid
            for txt_id in person[pid]['text']:
                self.txt2img[txt_id] = person[pid]['image']
                assert self.txt2pid[txt_id] == pid

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_text = self.annotation[index]['img_gen_text']

        img_caption = pre_caption(img_text, self.max_words)

        return image, index, img_caption


if __name__ == '__main__':
    tl,vl,tel = split_RSTP_PEDE()
    print(len(tl),len(vl),len(tel))
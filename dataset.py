import zipfile
import json
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Any
import pickle
import re
import constant
import numpy as np
import cv2
import torch
import random
import torchvision

with open("vocab.pkl", "rb") as f:
	VOCAB = pickle.load(f)

class ImageCaptioningDataset(Dataset):
	
	def __init__(self, image_zip_file, caption_zip_file, phase="train", caption_length=22):
		assert phase in ["train", "val"]
		self.map = {"train": 0, "val": 1}
		self.image_zip_file = image_zip_file
		self.caption_zip_file = caption_zip_file
		self.caption_length = caption_length

		zf = zipfile.ZipFile(caption_zip_file)
		with zf.open(zf.filelist[self.map[phase]].filename) as f:
			captions = json.loads(f.read())
		
		zf = zipfile.ZipFile(image_zip_file)
		zip_dict : Dict[str, zipfile.ZipInfo] = {} 
		for f in zf.filelist:
			if f.filename.endswith(".jpg"):
				filename = f.filename.strip("{}2014/".format(phase))
				zip_dict[filename] = f
		image_anno_dict = {}
		for anno in captions["annotations"]:
			image_id = anno["image_id"]
			caption = re.sub(r'[^\w\s]', '', anno["caption"].lower().strip())
			caption = self._convert_to_token(caption)
			if image_id not in image_anno_dict:
				image_anno_dict[image_id] = {}
				image_anno_dict[image_id]["captions"] = []
			image_anno_dict[image_id]["captions"].append(caption)
		for img_dict in captions["images"]:
			image_id = img_dict["id"]
			file_name = img_dict["file_name"]
			if image_id not in image_anno_dict:
				raise ValueError("The caption label file is broken.")
			image_anno_dict[image_id]["file_name"] = zip_dict[file_name]

		self.images : List[zipfile.ZipFile] = []
		self.captions : List[List[int]]= []
		self.zf = zf
		for k, v in image_anno_dict.items():
			self.images.append(v["file_name"])
			self.captions.append(v["captions"])
		assert len(self.images) == len(self.captions)

	def _convert_to_token(self, caption: str):
		"""
		Convert caption string to list of int tokens.
		"""
		words = caption.split(" ")
		words = words[:self.caption_length-2]
		tokens = []
		for word in words:
			if word not in VOCAB:
				tokens.append(VOCAB.index(constant.UNK))
				continue
			tokens.append(VOCAB.index(word))
		tokens = [VOCAB.index(constant.START)] + tokens +  [VOCAB.index(constant.END)]
		tokens = tokens + [VOCAB.index(constant.PAD)]*(self.caption_length-len(tokens))
		return tokens

	def _resize_img(self, img, shape=(300, 300)):
		h, w = img.shape[0], img.shape[1]
		pad_left = 0
		pad_right = 0
		pad_top = 0
		pad_bottom = 0
		if h > w:
			diff = h - w
			pad_top = diff - diff // 2
			pad_bottom = diff // 2
		else:
			diff = w - h
			pad_left = diff - diff // 2
			pad_right = diff // 2
		cropped_img = img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]
		cropped_img = cv2.resize(cropped_img, shape)
		return cropped_img

	def __getitem__(self, index):
		image = self.images[index]
		caption = self.captions[index]
		caption = caption[random.randint(0, len(caption)-1)]

		with self.zf.open(image) as f:
			buf = f.read()
			img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img = self._resize_img(img)
		img = torchvision.transforms.ToTensor()(img)
		caption = torch.as_tensor(caption)

		return img, caption

	def __len__(self):
		return len(self.images)
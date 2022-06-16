import config
import constant
import torch.optim as optim
from models import CNNEncoder, RNNDecoder
import torch
import torch.nn as nn
from dataset import ImageCaptioningDataset
from torch.utils.data.dataloader import DataLoader

class Predictor:

    def __init__(self, encoder : CNNEncoder, decoder : RNNDecoder, dataset: ImageCaptioningDataset = None):
        self.device = "cpu"
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        if dataset != None:
            self.dataloader = DataLoader(dataset, batch_size=config.TEST_BATCH_SIZE)

    def evaluate(self, img, caption):
        original_caption = caption
        print(original_caption)
        with torch.no_grad():
            img = img.to(self.device)
            caption = caption.to(self.device)
            end_indx = torch.argmax(1*(caption == constant.END_IND), dim=-1)
            caption = caption[:end_indx]
            img_embed = self.encoder(img)
            caption = self.decoder(caption, img_embed)
            caption = torch.argmax(caption, dim=-1)
            print(caption)
    
    def predict(self, img):
        with torch.no_grad():
            img = img.to(self.device)
            img_embed = self.encoder(img)
            caption = constant.START_IND
            caption = self.decoder(caption, img_embed, prediction=True)
            print(caption)
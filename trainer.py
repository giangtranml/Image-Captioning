from torch.utils.data.dataloader import DataLoader
import config
import torch.optim as optim
from models import CNNEncoder, RNNDecoder
import torch.nn as nn
from dataset import ImageCaptioningDataset
from tqdm import tqdm
import torch
from predictor import Predictor
import constant

class Trainer:

    def __init__(self, encoder : CNNEncoder, decoder : RNNDecoder, dataset : ImageCaptioningDataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE)
        self.optimizer = optim.Adam(decoder.parameters())
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        for e in range(config.EPOCH):
            pbar = tqdm(self.dataloader, desc="Epoch: {}".format(e+1))
            for i, (img, caption) in enumerate(pbar):
                img = img.to(self.device)
                caption = caption.to(self.device)
                img_embed = self.encoder(img)
                output = self.decoder(caption, img_embed)
                output = output.permute(0, 2, 1)
                loss = self.loss(output, caption)

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

                pbar.set_description(desc="Epoch " + str(e+1) + " - Loss: %.5f" % (loss.item()))
                
                if ((i+1)%50) == 0:
                    for img_, caption_ in zip(img, caption):
                        evaluator = Predictor(self.encoder, self.decoder)
                        evaluator.predict(img_.unsqueeze(dim=0))
                    self.encoder = self.encoder.to(self.device)
                    self.decoder = self.decoder.to(self.device)

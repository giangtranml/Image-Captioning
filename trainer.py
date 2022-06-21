from torch.utils.data.dataloader import DataLoader
import config
import torch.optim as optim
from models import CNNEncoder, RNNDecoder
import torch.nn as nn
from dataset import ImageCaptioningDataset
from tqdm import tqdm
import torch
from predictor import Predictor
import utils

class Trainer:

    def __init__(self, encoder : CNNEncoder, decoder : RNNDecoder, train_dataset : ImageCaptioningDataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        self.optimizer = optim.Adam(decoder.parameters())
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        for e in range(config.EPOCH):
            pbar = tqdm(self.train_dataloader, desc="Epoch: {}".format(e+1))
            for i, (img, caption) in enumerate(pbar):
                img = img.to(self.device)
                caption = caption.to(self.device)
                img_embed = self.encoder(img)
                output = self.decoder(caption[:, :-1], img_embed)
                output = output.permute(0, 2, 1)
                loss = self.loss(output, caption[:, 1:])

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

                pbar.set_description(desc="Epoch " + str(e+1) + " - Loss: %.5f" % (loss.item()))
                
                if ((i+1)%100) == 0:
                    predictor = Predictor(self.encoder, self.decoder)
                    output = predictor.predict(img[-1].unsqueeze(0))
                    print(utils.tokens_to_words(output))
                    self.encoder = self.encoder.to(self.device)
                    self.decoder = self.decoder.to(self.device)
        
        torch.save(self.decoder, "model_decoder.pth")
                    

    def evaluate(self):
        eval_pbar = tqdm(self.eval_dataloader, desc="Evaluation...")
        with torch.no_grad():
            for img, caption in eval_pbar:
                img = img.to(self.device)
                caption = caption.to(self.device)
                img_embed = self.encoder(img)
                output = self.decoder(caption[:, :-1], img_embed)
                output = output.permute(0, 2, 1)
                loss = self.loss(output, caption[:, 1:])

                eval_pbar.set_description(desc="Eval loss: %.5f" % (loss.item()))

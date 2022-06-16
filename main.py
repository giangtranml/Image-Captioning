from dataset import ImageCaptioningDataset
from models import CNNEncoder, RNNDecoder
import config
from trainer import Trainer
from predictor import Predictor
import torch

def train():
    dataset = ImageCaptioningDataset(image_zip_file="data/images_train.zip", caption_zip_file="data/captions.zip", phase="train")
    cnn = CNNEncoder()
    rnn = RNNDecoder(config.NUM_VOCAB)
    trainer = Trainer(cnn, rnn, dataset)
    trainer.train()

def test():
    dataset = ImageCaptioningDataset(image_zip_file="data/images_train.zip", caption_zip_file="data/captions.zip", phase="train")
    cnn = CNNEncoder()
    rnn = torch.load("model.pth")
    predictor = Predictor(cnn, rnn, dataset)
    predictor.predict(None)

def main():
    train()
    test()

if __name__ == "__main__":
    main()
from dataset import ImageCaptioningDataset
from models import CNNEncoder, RNNDecoder
import config
from trainer import Trainer
from predictor import Predictor
import torch

def train():
    train_dataset = ImageCaptioningDataset(image_zip_file="data/images_train.zip", caption_zip_file="data/captions.zip", phase="train")
    cnn = CNNEncoder()
    cnn.eval()
    rnn = RNNDecoder(config.NUM_VOCAB)
    trainer = Trainer(cnn, rnn, train_dataset)
    trainer.train()

def main():
    train()

if __name__ == "__main__":
    main()
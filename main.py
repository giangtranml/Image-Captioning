from dataset import ImageCaptioningDataset
from torch.utils.data.dataloader import DataLoader
from models import CNNEncoder, RNNDecoder
import constant

def main():
    dataset = ImageCaptioningDataset(image_zip_file="data/images_train.zip", caption_zip_file="data/captions.zip", phase="train")
    dataloader = DataLoader(dataset, batch_size=2)
    cnn = CNNEncoder()
    rnn = RNNDecoder(constant.NUM_VOCAB)
    for img, caption in dataloader:
        img_embed = cnn(img)
        out, (h, c) = rnn(caption, img_embed)
        print(out.shape)
        print(h.shape)
        print(c.shape)

if __name__ == "__main__":
    main()
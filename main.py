from generator.convolutional_block import ConvolutionalBlock
from dataset.custom_image_dataset import CustomImageDataset
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from train.checkpoints import load_checkpoint, save_checkpoint
from albumentations.pytorch import ToTensorV2
from train.train_model import train_model
from torch.utils.data import DataLoader
from descriminator.descriminator import Discriminator
from generator.generator import Generator
import numpy as np
from PIL import Image
import os
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np



def main():
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    photo_path = '././input-dataset/photo_jpg'
    monet_path = '././input-dataset/monet_jpg'
    
    CHECKPOINT_GENERATOR_PHOTO = "././trained-models/gen-photo.pth.tar"
    CHECKPOINT_GENERATOR_MONET = "././trained-models/gen-monet.pth.tar"
    CHECKPOINT_DISCRIMINATOR_PHOTO = "././trained-models/disc-photo.pth.tar"
    CHECKPOINT_DISCRIMINATOR_MONET= "././trained-models/disc-mnoet.tar"

    transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    )
    
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    NUM_WORKERS = 4
    NUM_EPOCHS = 10

    disc_photo = Discriminator(in_channels=3).to(DEVICE)
    disc_monet = Discriminator(in_channels=3).to(DEVICE)
    gen_photo = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_monet = Generator(img_channels=3, num_residuals=9).to(DEVICE)


    opt_disc = optim.Adam(
        list(disc_photo.parameters()) + list(disc_monet.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_photo.parameters()) + list(gen_monet.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    

    dataset = CustomImageDataset(
        photo_path,
        monet_path,
        transform=transforms,
    )
   
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        model_gen_monet, model_gen_photo = train_model(
            DEVICE,
            disc_photo,
            disc_monet,
            gen_monet,
            gen_photo,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

    save_checkpoint(gen_photo, opt_gen, filename=CHECKPOINT_GENERATOR_PHOTO)
    save_checkpoint(gen_monet, opt_gen, filename=CHECKPOINT_GENERATOR_MONET)
    save_checkpoint(disc_photo, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_PHOTO)
    save_checkpoint(disc_monet, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_MONET)
    
    photo_np = np.array(Image.open('./input-dataset/photo_jpg/00068bc07f.jpg').convert("RGB"))

    single_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    photo_tensor = single_transform(photo_np) 
    photo_tensor = photo_tensor.to(DEVICE)

    monet_pic = gen_monet(photo_tensor)
    save_image(monet_pic * 0.5 + 0.5, f"./output-dataset/monet_jpg/generated_monet_00068bc07f.jpg")



if __name__ == "__main__":
    main()
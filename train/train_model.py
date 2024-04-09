import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


zip_filename = '././output-dataset/monet/imgaes.zip'

def add_files_to_zip(file_to_add):
    with zipfile.ZipFile(zip_filename, "a") as zip_ref:
        zip_ref.write(file_to_add)
        zip_ref.close()
    os.remove(file_to_add)


def train_model(device,
    discriminator_photo, discriminator_monet, generator_photo, generator_monet, 
    loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    Photo_real = 0
    Photo_fake = 0
    LAMBDA_ADVERS = 1
    LAMBDA_IDENTITY = 0.5
    LAMBDA_CYCLE = 10
    
    loop = tqdm(loader, leave=True)
    print(loop)
    count = 0
    
    # Train Descriminator
    for idx, (photo, monet) in enumerate(loop):
        count+=1
        print(f'@@@@@@ Count {count}')
        photo = photo.to(device)
        monet = monet.to(device)

        with torch.cuda.amp.autocast():
            fake_photo = generator_photo(monet)
            D_photo_real = discriminator_photo(photo)
            D_photo_fake = discriminator_photo(fake_photo.detach())
            Photo_real += D_photo_real.mean().item()
            Photo_fake += D_photo_fake.mean().item()
            D_Photo_real_loss = mse(D_photo_real, torch.ones_like(D_photo_real))
            D_Photo_fake_loss = mse(D_photo_fake, torch.zeros_like(D_photo_fake))
            D_Photo_loss = D_Photo_real_loss + D_Photo_fake_loss

            fake_monet = generator_monet(photo)
            D_monet_real = discriminator_monet(monet)
            D_monet_fake = discriminator_monet(fake_monet.detach())
            D_monet_real_loss = mse(D_monet_real, torch.ones_like(D_monet_real))
            D_monet_fake_loss = mse(D_monet_fake, torch.zeros_like(D_monet_fake))
            D_monet_loss = D_monet_real_loss + D_monet_fake_loss

            D_loss = (D_Photo_loss + D_monet_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators
        with torch.cuda.amp.autocast():
            # adversarial losses
            D_photo_fake = discriminator_photo(fake_photo)
            D_monet_fake = discriminator_monet(fake_monet)
            loss_G_Photo = mse(D_photo_fake, torch.ones_like(D_photo_fake))
            loss_G_Monet = mse(D_monet_fake, torch.ones_like(D_monet_fake))
            
            loss_advers = loss_G_Photo + loss_G_Monet
            

            # cycle losses
            cycle_monet = generator_monet(fake_photo)
            cycle_photo = generator_photo(fake_monet)
            cycle_monet_loss = l1(monet, cycle_monet)
            cycle_photo_loss = l1(photo, cycle_photo)
            
            loss_cycle= cycle_monet_loss + cycle_photo_loss
                
            
            # identity losses
            identity_monet = generator_monet(monet)
            identity_photo = generator_photo(photo)
            identity_monet_loss = l1(monet, identity_monet)
            identity_photo_loss = l1(photo, identity_photo)
            
            loss_identity= identity_monet_loss + identity_photo_loss
            
        
            # total loss
            generator_loss = (loss_advers * LAMBDA_ADVERS + loss_cycle * LAMBDA_CYCLE + loss_identity * LAMBDA_IDENTITY)
               

        opt_gen.zero_grad()
        g_scaler.scale(generator_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_photo * 0.5 + 0.5, f"././output-dataset/photo/photo_{idx}.jpg")
            save_image(fake_monet * 0.5 + 0.5, f"././output-dataset/monet/monet_{idx}.jpg")

        loop.set_postfix(Photo_real=Photo_real / (idx + 1), Photo_fake=Photo_fake / (idx + 1))
    return generator_monet, generator_photo
        
        
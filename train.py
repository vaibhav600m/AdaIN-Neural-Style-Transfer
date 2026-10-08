import sys
import os
from pathlib import Path

# Project root path setup for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

# Local imports
from utils.utils import *
from utils.models import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Local Paths (Update as per your local folder structure)
    parser.add_argument('--content_dir', type=str, default='data/content_data')
    parser.add_argument('--style_dir', type=str, default='data/style_data')
    parser.add_argument('--vgg', type=str, default='weights/vgg_normalised.pth')
    parser.add_argument('--experiment', type=str, default='local_run')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--final_size', type=int, default=256)
    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)
    parser.add_argument('--crop', action='store_true', default=True)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5) 

    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    # Loss Weights
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--tv_weight', type=float, default=1e-5)
    
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=4)
    
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--decoder_path', type=str, default=None)
    parser.add_argument('--optimizer_path', type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # MacBook MPS Support added here
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print("Using device:", device)
    
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch))
    
    if args.resume and args.decoder_path:
        print(f"Loading weights from {args.decoder_path}...")
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        if args.optimizer_path and os.path.exists(args.optimizer_path):
            optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))

    mse_loss = torch.nn.MSELoss()
    encoder.eval()
    num_batches = min(len(content_dataloader), len(style_dataloader))

    for epoch in range(args.start_epoch, args.epochs):
        current_epoch_num = epoch + 1
        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=num_batches)

        for content_batch, style_batch in progress_bar:
            content_batch, style_batch = content_batch.to(device), style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            g = decoder(t)
            g_feats = encoder(g)

            # Content Loss
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            # Multi-layer Style Loss
            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            # Total Variation (TV) Loss
            loss_tv = calc_tv_loss(g) * args.tv_weight

            loss = loss_c + (loss_s * args.style_weight) + loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Epoch {current_epoch_num}/{args.epochs} - Loss:{loss.item():.4f}')

        scheduler.step()

        if current_epoch_num % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{current_epoch_num}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{current_epoch_num}.pth')
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{current_epoch_num}.png', nrow=args.batch_size)

if __name__ == '__main__':
    main()
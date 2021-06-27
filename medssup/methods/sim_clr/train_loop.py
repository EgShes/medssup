from argparse import Namespace

import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from medssup.data.preprocessing import Preprocessing
from medssup.methods.sim_clr.model import SimCLRViT
from medssup.methods.sim_clr.optim_utils import (
    SimCrossEntropyLoss,
    evaluate,
    load_optimizer,
    train_iter,
)
from medssup.methods.sim_clr.training_utils import (
    get_augmentations,
    get_loaders,
    write_attention_maps,
)


def train_model(args: Namespace):
    device = torch.device(args.device)
    writer = SummaryWriter()

    preprocessing = Preprocessing(args.windows, (args.image_size, args.image_size))
    augmentations = get_augmentations()

    train_loader, val_loader, test_loader = get_loaders(args, preprocessing, augmentations)

    model = SimCLRViT.from_scratch(
        image_size=args.image_size,
        patch_size=args.patch_size,
        trf_depth=args.trf_depth,
        trf_heads=args.trf_heads,
        trf_dropout=args.trf_dropout,
        trf_dim=args.trf_dim,
        trf_emb_dropout=args.trf_emb_dropout,
        projection_dim=args.projection_dim,
    )
    model.to(device)

    optimizer, scheduler = load_optimizer(model, args.optimizer_type, args.lr, args.iterations, args.wd)
    criterion = SimCrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    scaler = GradScaler(enabled=args.fp16)

    best_loss = float("inf")
    for iteration, batch in tqdm(enumerate(train_loader), desc="Training", total=args.iterations):
        if iteration == args.iterations:
            print("Stopping")
            break
        train_loss = train_iter(model, batch, optimizer, criterion, scaler, device)
        writer.add_scalar("train_loss", train_loss, iteration)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)
        scheduler.step()

        if (iteration + 1) % args.eval_interval == 0:
            metrics = evaluate(model, val_loader, criterion, scaler, device)
            write_attention_maps(model.transformer, val_loader, writer, device)
            writer.add_scalars("eval", metrics, iteration)

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                checkpoint = {"model_state_dict": model.state_dict(), "arguments": args, "iteration": iteration}
                torch.save(checkpoint, f"{args.task_name}_{iteration}.pth")
            print(metrics)

"""
rqvae trainer
"""
import gin
import os
import numpy as np
import time
import wandb

from accelerate import Accelerator
from generative_recommenders.data.amazon import AmazonReviewsItemDataset
from generative_recommenders.data.utils import cycle
from generative_recommenders.models.rqvae import RqVae, QuantizeForwardMode
from generative_recommenders.modules.utils import parse_config
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DistributedSampler
from tqdm import tqdm


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/amazon",
    dataset=AmazonReviewsItemDataset,
    pretrained_rqvae_path=None,
    save_dir_root="out/rqvae/amazon/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    wandb_project="rqvae_training",
    wandb_log_interval=100,
    do_eval=True,
    mixed_precision_type="fp16",
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
    vae_sim_vq=False,
    vae_n_layers=3,
    encoder_model_name="./models_hub/sentence-t5-xl"
):
    """
    train rqvae
    """
    if wandb_logging:
        params = locals()

    # setup accelerator
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    train_dataset = dataset(root=dataset_folder, train_test_split="train", encoder_model_name=encoder_model_name)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        collate_fn=lambda batch: torch.tensor(batch, dtype=torch.float32),
        num_workers=16,
        pin_memory=True,
        prefetch_factor=10,
        persistent_workers=True)

    train_dataloader = cycle(train_dataloader)

    if do_eval:
        eval_dataset = dataset(root=dataset_folder, train_test_split="eval", encoder_model_name=encoder_model_name)
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.float32),
            num_workers=16,
            pin_memory=True,
            prefetch_factor=10,
            persistent_workers=True)
    
    train_dataloader = accelerator.prepare(train_dataloader)

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        codebook_last_layer_mode=vae_codebook_last_layer_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=wandb_project,
            config=params
        )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    with tqdm(
        initial=start_iter, 
        total=start_iter + iterations, 
        disable=not accelerator.is_main_process
    ) as pbar:
        losses = [[], [], []]
        for iter in range(start_iter, start_iter + 1 + iterations):        
            model.train()
            t = 0.2

            if iter == 0 and use_kmeans_init:
                buf = []
                seen = 0
                want = 20000
                while seen < want:
                    data = next(train_dataloader).to(device)
                    buf.append(data)
                    seen += data.size(0)
                big_batch = torch.cat(buf, dim=0)
                model(big_batch, t)

            optimizer.zero_grad()
            data = next(train_dataloader).to(device)

            with accelerator.autocast():
                model_output = model(data, gumbel_t=t)
                loss = model_output.loss

            accelerator.backward(loss)

            losses[0].append(loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process and wandb_logging:
                # Compute logs depending on training model_output here to avoid cuda graph overwrite from eval graph.
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                emb_norms_avg_log = {
                    f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item() for i in range(vae_n_layers)
                }
                train_log = {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "total_loss": loss.cpu().item(),
                    "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                    "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                    **emb_norms_avg_log,
                }

            if do_eval and ((iter + 1) % eval_every == 0 or iter + 1 == iterations):
                model.eval()
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=True) as pbar_eval:
                    eval_losses = [[], [], []]
                    for batch in pbar_eval:
                        data = batch.to(device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)

                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())
                    
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    
            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"/checkpoint_{iter}.pt")

                if wandb_logging and (iter + 1) % wandb_log_interval == 0:
                    wandb.log({
                        **train_log,
                        **id_diversity_log
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()
    

if __name__ == "__main__":
    parse_config()
    train()
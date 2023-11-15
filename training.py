# I based the embedding learning on this repo 
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb

import os
import tqdm
import random
import argparse
import numpy as np

import torch
import torch.linalg as linalg
import torch.utils.data as data
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDPMScheduler

from core.dataset import get_dataset, TextualDataset
from core.utils import Print, add_new_tokens, load_tokens_and_embeddings, save_tokens_and_embeddings
from core.phrases import get_phrase_generator


def freeze(m, names):
    for n in names:
        for p in getattr(m, n).parameters():
            p.requires_grad = False


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_model', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--embedding-files', type=str, nargs='+', default=[])

    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--data_dir', type=str, default='/home/2017025/gjeann01/save/celeba')
    parser.add_argument('--partition', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='CelebAHQ')
    parser.add_argument('--seed', type=int, default=99999999)
    
    # classifier arguments
    parser.add_argument('--label_query', type=int, default=31)
    parser.add_argument('--training_label', type=int, default=-1,
                        help='Only used for binary classification')

    # token related args
    parser.add_argument('--custom_tokens', type=str, nargs='+', required=True)
    parser.add_argument('--custom_tokens_init', type=str, nargs='+', required=True)
    parser.add_argument('--phase', type=str, default='context',
                        choices=['context', 'class'])
    parser.add_argument('--base_prompt', type=str, default='A picture of',
                        help='Used only in the "class" phase. It will be the base of the phrase to train the model')
    parser.add_argument('--enable_xformers_memory_efficient_attention',
                        action='store_true')

    return parser.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():

    # =================================================================
    # Custom variables
    args = arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    device = torch.device('cuda')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # =================================================================
    # Instantiate Pipeline
    Print('Initializing Stable Diffusion')
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch_dtype,
    )
    pipeline.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")

    # =================================================================
    # Initialize token(s)

    # load previous tokens
    load_tokens_and_embeddings(
        sd_model=pipeline,
        files=args.embedding_files
    )

    # generate new tokens
    index_no_updates, placeholder_token_ids, initializer_token_ids = add_new_tokens(
        new_tokens=args.custom_tokens,
        init_tokens=args.custom_tokens_init,
        sd_model=pipeline,
        dtype=torch_dtype
    )

    # =================================================================
    # Get tokenizer and text encoder

    text_encoder = pipeline.text_encoder
    text_encoder.requires_grad_(True)
    tokenizer = pipeline.tokenizer

    # freeze all network parameters except the embeddings
    freeze(pipeline, ['vae', 'unet'])
    freeze(pipeline.text_encoder.text_model, ['encoder', 'final_layer_norm'])
    freeze(text_encoder.text_model.embeddings, ['position_embedding'])
    text_encoder.get_input_embeddings().weight.requires_grad = True

    if args.enable_xformers_memory_efficient_attention:
        '''
        Must have 0.0.17>xformer.__version__
        '''
        pipeline.unet.enable_xformers_memory_efficient_attention()

    # =================================================================
    # Optimizer
    Print('Initializing optimizer and dataset')
    # Here, there is a problem, we are optimizing the other embeddings as well with the weight decay as well.
    # We will compute the wd manually
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0,  # manually implemented
        eps=args.adam_epsilon,
    )

    # =================================================================
    # Dataset
    dataset = get_dataset(args)
    dataset = TextualDataset(
        custom_tokens=args.custom_tokens,
        base_prompt_generator=get_phrase_generator(args),
        dataset=dataset
    )
    loader = data.DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=5,
                             shuffle=True,
                             worker_init_fn=seed_worker,
                             generator=g)

    # =================================================================
    # Training Loop

    num_chunks = args.batch_size // args.mini_batch_size

    Print('Training!')
    differences = []
    iterations = 0

    def data_loader(iterations):
        while True:
            yield from loader

    for image, text in tqdm.tqdm(data_loader(iterations), desc='Iterations', total=args.iterations):
        iterations += 1
        image = image.to(device, dtype=torch_dtype)
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device, dtype=torch.long)
        B = image.size(0)

        for img, text_ids in zip(image.chunk(num_chunks), text_input_ids.chunk(num_chunks)):
            # Image encoding
            with torch.no_grad():
                latents = pipeline.vae.encode(img).latent_dist.sample().detach()  # this encodes 256x256 images!
                latents = latents * pipeline.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = pipeline.text_encoder(text_ids)[0].to(dtype=torch_dtype)

            # Predict the noise residual
            model_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="sum") / B
            loss.backward()

        # modify gradients of tokens that we don't want to change
        with torch.no_grad():
            grad = text_encoder.get_input_embeddings().weight.grad
            # weight decay
            grad = grad.add(text_encoder.get_input_embeddings().weight.data,
                            alpha=args.weight_decay)
            # zero-out gradients of those embeddings we don't want to modify
            grad = grad * (1 - index_no_updates.unsqueeze(1))
            text_encoder.get_input_embeddings().weight.grad = grad

        optimizer.step()
        optimizer.zero_grad()
        
        if (iterations % 500 == 0) or (iterations > args.iterations):
            with torch.no_grad():
                embeddings = text_encoder.get_input_embeddings().weight.data
                d = (embeddings[placeholder_token_ids] - embeddings[initializer_token_ids]).abs().mean(dim=1, keepdim=True)
                differences.append(d.detach().cpu())
                Print(f'Mean difference at iteration {iterations}:', differences[-1])
            save_tokens_and_embeddings(
                sd_model=pipeline,
                tokens=args.custom_tokens,
                output=args.output_path[:-4] + f'-ckpt-{iterations}.pth',
            )

        if iterations > args.iterations:
            break

    pipeline.to('cpu')
    save_tokens_and_embeddings(
        sd_model=pipeline,
        tokens=args.custom_tokens,
        output=args.output_path,
    )

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    differences = torch.cat(differences, dim=1).numpy()
    for idx, token in enumerate(args.custom_tokens):
        plt.plot(differences[idx, :], label=token)
    plt.title('L_1 difference')
    plt.legend()
    plt.savefig(f'differences-{args.phase}.png')
    plt.close()


if __name__ == '__main__':
    main()

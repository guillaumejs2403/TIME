import os
import os.path as osp
import time
import tqdm
import copy
import argparse
import itertools
import numpy as np
import PIL.Image as Image

import torch
import torch.utils.data as TD
import torch.nn.functional as F

from diffusers import DDIMInverseScheduler, DDIMScheduler

from core.edict import EDICT
from core.utils import (
    Print,
    accuracy,
    ImageSaver,
    merge_chunks,
    get_prediction,
    generate_prompt,
    load_tokens_and_embeddings
)
from core.dataset import (
    ChunkedDataset,
    get_dataset,
    BINARYDATASET,
    MULTICLASSDATASETS,
    SlowSingleLabel
)

from models import get_classifier


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)

    # SD model
    parser.add_argument('--sd_model', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--sd_image_size', type=int, default=512)
    parser.add_argument('--pos_custom_token', type=str, default='|<A*P>|')
    parser.add_argument('--neg_custom_token', type=str, default='|<A*N>|')
    parser.add_argument('--custom_token', type=str, default='|<A*>|')
    parser.add_argument('--custom_obj_token', type=str, default='|<C*>|')
    parser.add_argument('--base_prompt', type=str, default='A |<C*>| picture')
    parser.add_argument('--embedding_files', type=str, nargs='+', default=[])

    # Sampling Hyperparameters
    parser.add_argument('--num_inference_steps', type=int, default=[35], nargs='+')
    parser.add_argument('--total_num_inference_steps', type=int, default=50)
    parser.add_argument('--p', type=float, default=0.93)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--use_negative_guidance_denoise', action='store_true')
    parser.add_argument('--use_negative_guidance_inverse', action='store_true')
    parser.add_argument('--guidance-scale-denoising', type=float, default=[5], nargs='+')
    parser.add_argument('--guidance-scale-invertion', type=float, default=[5], nargs='+')

    # dataset params
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--label_query', type=int, default=31)
    parser.add_argument('--label_target', type=int, default=-1)
    parser.add_argument('--data_dir', type=str, default='/home/personnels/jeanner211/DATASETS/celeba')
    parser.add_argument('--dataset', type=str, default='CelebAHQ')
    parser.add_argument('--batch_size', type=int, default=1)

    # classifier params
    parser.add_argument('--classifier_image_size', type=int, default=256)
    parser.add_argument('--classifier_path', type=str, default='/home/personnels/jeanner211/RESULTS/HQCelebA/classifier/checkpoint.tar')

    # Others
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--chunks', type=int, default=1)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--num_samples', type=int, default=9999999999999999)
    parser.add_argument('--merge-chunks', action='store_true')
    parser.add_argument('--enable_xformers_memory_efficient_attention',
                        action='store_true')
    return parser.parse_args()


def tensor_to_numpy(img):
    img = (img * 255).to(dtype=torch.uint8).detach().cpu()
    return img.permute(0, 2, 3, 1).numpy()


def main():

    torch.set_grad_enabled(False)

    # =================================================================
    # Custom variables

    args = arguments()
    print(args)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    device = torch.device('cuda')
    torch.manual_seed(4)
    np.random.seed(4)

    # ================================================================
    # Merge All results

    if args.merge_chunks:
        merge_chunks(args)
        return

    # ================================================================
    # Load classifier

    Print('Loading classifier to check if we flipped the image')
    classifier = get_classifier(args)
    classifier.eval()
    classifier.to(device)

    # =================================================================
    # Instantiate Pipeline
    Print('Loading pipeline')
    pipeline = EDICT.from_pretrained(
        args.sd_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    load_tokens_and_embeddings(sd_model=pipeline, files=args.embedding_files)
    pipeline.reverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    pipeline.text_encoder.eval()  # just in case
    pipeline.set_progress_bar_config(disable=True)  # disable anoying progress bar

    if args.enable_xformers_memory_efficient_attention:
        '''
        Must have 0.0.17>xformer.__version__
        '''
        pipeline.unet.enable_xformers_memory_efficient_attention()


    # =================================================================
    # Loading Dataset

    args.image_size = args.classifier_image_size if args.dataset in ['BDD100k'] else args.sd_image_size
    dataset, postprocess = get_dataset(args, training=False)

    filter_label = -1
    if args.label_target != -1:
        filter_label = 1 - args.label_target \
                 if args.dataset in BINARYDATASET else args.label_query

    dataset = SlowSingleLabel(label=filter_label,
                              dataset=dataset,
                              maxlen=args.num_samples)

    dataset = ChunkedDataset(dataset,
                             chunk=args.chunk,
                             num_chunks=args.chunks)

    loader = TD.DataLoader(dataset,
                           batch_size=args.batch_size,
                           num_workers=5)

    # ========================================
    # Get variables of interest

    start_time = time.time()

    stats = {
        'cf': 0,
        'cf5': 0,
        'l1': 0,
        'l inf': 0,
        'p': 0
    }

    stats = {
        'n': 0,
        'clean acc': 0,
        'clean acc5': 0,
        'explanation': copy.deepcopy(stats),
    }

    image_saver = ImageSaver(
        output_path=args.output_path,
        exp_name=args.exp_name,
    )

    # set guidance_scale_invertion, guidance_scale_denoising and num_inference_steps to the same length
    max_length = max(len(args.guidance_scale_invertion), len(args.guidance_scale_denoising), len(args.num_inference_steps))
    if args.guidance_scale_invertion == 1:
        args.guidance_scale_invertion *= max_length

    if args.guidance_scale_denoising == 1:
        args.guidance_scale_denoising *= max_length

    if args.num_inference_steps == 1:
        args.num_inference_steps *= max_length

    assert len(args.guidance_scale_invertion) == max_length
    assert len(args.guidance_scale_denoising) == max_length
    assert len(args.num_inference_steps) == max_length

    for idx, (indexes, img, lab) in enumerate(loader):
        print(f'[Chunks ({args.chunk}+1) / {args.chunks}] {idx} / {len(loader)} | Time: {int(time.time() - start_time)}s')

        B = img.size(0)
        img = img.to(device, dtype=torch_dtype)
        lab = lab.to(device, dtype=torch_dtype if args.dataset in BINARYDATASET else torch.long)

        img_orig = postprocess(img, args.classifier_image_size)
        c_log, c_pred = get_prediction(classifier, img_orig,
                                       args.dataset in BINARYDATASET)

        (acc1, acc5), _ = accuracy(c_log, lab, binary=args.dataset in BINARYDATASET)
        stats['clean acc'] += acc1.sum().item()
        stats['clean acc5'] += acc5.sum().item()
        stats['n'] += lab.size(0)

        # =================================================================
        # construct target

        target = None
        if args.label_target != -1:
            target = torch.ones_like(lab) * args.label_target
            target[lab != c_pred] = lab[lab != c_pred]
        elif args.dataset in BINARYDATASET:
            target = 1 - c_pred
            target[lab != c_pred] = lab[lab != c_pred]

        # =================================================================
        # Generate phrases

        target_prompts, source_prompts = [], []
        for t, p in zip(target, c_pred):
            n, t = generate_prompt(
                args=args,
                target=t.item(),
                pred=p.item(),
                binary=args.dataset in BINARYDATASET,
            )
            target_prompts.append(t)
            source_prompts.append(n)

        # =================================================================
        # CE generation
        transformed = torch.zeros_like(lab).bool()
        
        for jdx, (gsi, gsd, nis) in enumerate(zip(args.guidance_scale_invertion,
                                             args.guidance_scale_denoising,
                                             args.num_inference_steps)):

            tp = [t for i, t in enumerate(target_prompts) if not transformed[i].item()]
            sp = [s for i, s in enumerate(source_prompts) if not transformed[i].item()]

            # Inversion
            xt, yt, _, _, feats = pipeline.Invert(
                prompt=source_prompts,
                prompt_embeds=None,
                image=img[~transformed],
                num_inference_steps=nis,
                total_num_inference_steps=args.total_num_inference_steps,
                guidance_scale=gsi,
                p=args.p,
                return_pil=False,
                negative_prompt=tp if args.use_negative_guidance_inverse else None,
                l2=args.l2,
            )

            rec = None
            if args.recover:
                rec = pipeline.Denoise(
                    prompt=source_prompts,
                    prompt_embeds=None,
                    xt=xt,
                    yt=yt,
                    num_inference_steps=nis,
                    total_num_inference_steps=args.total_num_inference_steps,
                    guidance_scale=gsi,
                    p=args.p,
                    return_pil=False,
                    negative_prompt=tp if args.use_negative_guidance_inverse else None,
                    l2=args.l2,
                    feats=feats
                )
                rec = postprocess(rec, args.classifier_image_size)

            # Denoising
            cf = pipeline.Denoise(
                prompt=target_prompts,
                prompt_embeds=None,
                xt=xt,
                yt=yt,
                num_inference_steps=nis,
                total_num_inference_steps=args.total_num_inference_steps,
                guidance_scale=gsd,
                p=args.p,
                return_pil=False,
                negative_prompt=sp if args.use_negative_guidance_denoise else None,
                l2=args.l2,
                feats=feats
            )
            cf = postprocess(cf, args.classifier_image_size)

            # check if explanation flipped
            log, _ = get_prediction(
                classifier, cf,
                binary=args.dataset in BINARYDATASET
            )
            (flipped,), _ = accuracy(
                log, target[~transformed],
                binary=args.dataset in BINARYDATASET,
                topk=(1,)
            )

            if jdx == 0:
                ce = cf.clone().detach()
            ce[~transformed] = cf
            transformed[~transformed] = flipped
            if transformed.float().sum().item() == transformed.size(0):
                break

        # =================================================================
        # Checking CE Accuray

        ce_log, ce_pred = get_prediction(
            classifier, ce,
            binary=args.dataset in BINARYDATASET
        )
        (cf, cf5), prob = accuracy(
            ce_log, target,
            binary=args.dataset in BINARYDATASET
        )
        l1 = (img_orig - ce).view(B, -1).abs().mean(dim=1).cpu()
        linf = (img_orig - ce).abs().view(B, -1).max(dim=1)[0].cpu()
        stats['explanation']['cf'] += cf.sum().item()
        stats['explanation']['cf5'] += cf5.sum().item()
        stats['explanation']['l1'] += l1.sum().item()
        stats['explanation']['l inf'] += linf.sum().item()
        stats['explanation']['p'] += prob[cf].sum().item()

        # save images
        image_saver(
            imgs=tensor_to_numpy(img_orig),
            cfs=tensor_to_numpy(ce),
            recs=None if not args.recover else tensor_to_numpy(rec),
            target=target,
            label=lab,
            pred=c_pred,
            pred_cf=ce_pred,
            l_inf=linf,
            l_1=l1,
            indexes=indexes.numpy()
        )

        if (idx + 1) == len(loader):
            print(f'[Chunks ({args.chunk}+1) / {args.chunks}] {idx + 1} / {len(loader)} | Time: {int(time.time() - start_time)}s')
            print('\nDone')
            break

    stats['clean acc'] /= stats['n']
    stats['clean acc5'] /= stats['n']
    correct = stats['explanation']['cf']
    for k in stats['explanation'].keys():
        if k == 'p':
            stats['explanation'][k] /= correct
            continue
        stats['explanation'][k] /= stats['n']

    prefix = '' if args.chunks == 1 else f'c-{args.chunk}_{args.chunks}-'
    with open(osp.join(args.output_path, 'Results', args.exp_name, prefix + 'summary.yaml'), 'w') as f:
        f.write(str(stats))


if __name__ == '__main__':
    main()

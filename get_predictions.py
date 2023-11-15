import torch
import torch.utils.data as data

from core.dataset import CelebAHQDataset, BDD100k
from models import get_classifier

import os
import tqdm
import argparse
import pandas as pd


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True,
                        choices=['BDD', 'CelebAHQ'],
                        help='Dataset name')
    parser.add_argument('--partition', type=str, default='train',
                        help='Dataset partition')
    parser.add_argument('--data-dir', required=True, type=str,
                        help='dataset path')
    parser.add_argument('--label-query', type=int, default=0,
                        help='Query label to check. Only applies for binary datasets')
    parser.add_argument('--image-size', default=256, type=int,
                        help='dataset image size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Inference batch size')
    parser.add_argument('--classifier-path', required=True,
                        help='path to classifier')

    return parser.parse_args()


if __name__ == '__main__':

    torch.set_grad_enabled(False)

    args = arguments()
    device = torch.device('cuda:0')
    os.makedirs('utils', exist_ok=True)

    if args.dataset == 'CelebAHQ':
        dataset = CelebAHQDataset(
            image_size=args.image_size,
            data_dir=args.data_dir,
            partition=args.partition,
            normalize=False,
            random_crop=False,
            random_flip=False,
            return_filename=True,
            label_query=args.label_query
        )
    elif args.dataset == 'BDD':
        dataset = BDD100k(
            image_size=256,
            data_dir=args.data_dir,
            partition=args.partition,
            normalize=False,
            padding=False
        )

    loader = data.DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=5,
                             shuffle=False)

    classifier = get_classifier(args)
    classifier.to(device).eval()

    d = {'idx': [],
         'prediction': []}
    n = 0
    acc = 0

    for img, lab, img_file in tqdm.tqdm(loader):

        img = img.to(device)
        lab = lab.to(device)
        pred = (classifier(img) > 0).int()
        acc += (pred == lab).float().sum().item()
        n += lab.size(0)

        d['prediction'] += [p.item() for p in pred]
        d['idx'] += list(img_file)

    print(acc / n)
    df = pd.DataFrame(data=d)
    df.to_csv(
        'utils/{}-{}-prediction-label-{}.csv'.format(args.dataset.lower(), args.partition, args.label_query),
        index=False
    )

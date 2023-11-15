import os
import os.path as osp
import random
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm

# ============================================================================
# Variable for binary and multiclass datasets
# ============================================================================

BINARYDATASET = ['CelebA', 'CelebAHQ', 'CelebAMV', 'BDDOIA', 'BDD100k', 'BDD']
MULTICLASSDATASETS = ['ImageNet']


# ============================================================================
# get_dataset function
# ============================================================================


def get_dataset(args, normalize=True, training=True):
    if args.dataset == 'CelebAHQ':
        dataset_kwargs = {
            'image_size': args.image_size,
            'data_dir': args.data_dir,
            'partition': args.partition,
            'normalize': normalize,
            'label_query': args.label_query,
            'random_crop': False,
        }
        if training:
            dataset = CHQFilter(
                filter_by=args.training_label,
                negate=False,
                **dataset_kwargs
            )

        else:
            dataset_kwargs['random_flip'] = False
            dataset = (CelebAHQDataset(**dataset_kwargs), celebahq_postprocessing)

    elif args.dataset == 'BDD100k':
        dataset_kwargs = {
            'image_size': args.image_size,
            'data_dir': args.data_dir,
            'partition': args.partition,
            'label_query': args.label_query,
            'normalize': normalize,
            'padding': True,
        }
        if training:
            dataset = FilteredBDD100k(
                filter_by=args.training_label,
                negate=False,
                **dataset_kwargs
            )
        else:
            dataset = (BDD100kForCE(**dataset_kwargs), bdd_postprocessing)

    elif args.dataset == 'ImageNet':
        pass

    return dataset


# ============================================================================
# Postprocessing functions
# ============================================================================


def celebahq_postprocessing(img, size):
    img = F.interpolate(img, size=size, mode='bilinear')
    img = (img + 1) / 2
    return img.clamp(0, 1)


def bdd_postprocessing(img, size):
    half = size // 2
    img = img[:, :, half:(size + half), :]
    img = (img + 1) / 2
    return img.clamp(0, 1)


# ============================================================================
# Chunked dataset
# ============================================================================


class ChunkedDataset:
    def __init__(self, dataset, chunk=0, num_chunks=1):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % num_chunks) == chunk]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        i = [self.indexes[idx]]
        i += list(self.dataset[i[0]])
        return i


# ============================================================================
# Datasets
# ============================================================================


class CelebAHQDataset():
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        label_query=-1,
        normalize=True,
        return_filename=False
    ):
        from io import StringIO
        # read annotation files
        with open(osp.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt'), 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' +  datastr.replace('  ', ' ')

        with open(osp.join(data_dir, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']

        mapstr = ' '.join(mapstr)

        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        mapping_df.rename(columns={'orig_file': 'image_id'}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on='image_id')

        self.data_dir = data_dir

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ])

        self.query = label_query
        self.class_cond = class_cond
        self.return_filename = return_filename

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        # choose prediction
        img_file = sample['idx']
        label = sample[2:].to_numpy()
        if self.query != -1:
            label = label[self.query].item()
        else:
            label = torch.from_numpy(label)

        with open(osp.join(self.data_dir, 'CelebA-HQ-img', img_file), "rb") as f:
            img = Image.open(f)
            img_PIL = img.convert('RGB')

        img = self.transform(img_PIL)

        if self.return_filename:
            return img, label, img_file

        return img, label


class CHQFilter(CelebAHQDataset):
    def __init__(self, filter_by, negate, **kwargs):
        '''
        params
            :filter_by: choosen label to filter the dataset. If set to -1, it does
                        not removes anything (basically is an identity)
            :negate: If set to true, KEEPS all datapoints with the "filter_by" input.
                     If false, REMOVES all datapoints
        '''
        super().__init__(**kwargs)
        self.predictions = pd.read_csv('utils/celebahq-{}-prediction-label-{}.csv'.format(kwargs['partition'], kwargs['label_query']))

        # filter data by the predictions
        self.filter_by = filter_by
        if filter_by == -1:
            return

        # just in case of a mismatch
        merged = pd.merge(self.data, self.predictions, on='idx')
        cond = merged['prediction'] == self.filter_by
        if negate:
            cond = ~cond
        self.data = merged[cond]


# ============================================================================
# BDD
# ============================================================================


class BDD100k():  # Jacob et al trained on 10k subset of BDD100k. datadir ~/save/BDD/images/10k/. Partition train
    def __init__(
        self,
        data_dir,
        image_size,
        partition,
        padding=False,
        normalize=True,
        label_query=None  # ignored, but used for consistency
    ):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, 2 * image_size)),
            transforms.ToTensor(),
            torch.nn.ConstantPad2d((0, 0, image_size // 2, image_size // 2), 0) if padding else lambda x: x,  # lambda x: F.pad(x, (0, 0, 128, 128), value=0)
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.data_dir = data_dir
        self.partition = partition
        self.root = osp.join(self.data_dir, self.partition)
        self.items = os.listdir(self.root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name = self.items[idx]
        with open(osp.join(self.root, name), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return self.transform(img), 0, name


class BDD100kForCE(BDD100k):
    def __getitem__(self, idx):
        img = super().__getitem__(idx)[0]
        return img, 0


class FilteredBDD100k(BDD100k):
    def __init__(self, filter_by=-1, negate=False, **kwargs):
        super().__init__(**kwargs)
        if filter_by == -1:
            return

        predictions = pd.read_csv('utils/bdd100k-{}-prediction-label-{}.csv'.format(kwargs['partition'], kwargs['label_query']))
        operator = (lambda x, y: x != y) if negate else (lambda x, y: x == y)
        
        data = predictions['idx'][operator(predictions['prediction'], filter_by)]
        # filter names, might be a little brute forced :P
        self.items = [x for x in filter(lambda x: data.eq(x).any(), self.items)]


# ============================================================================
# ImageNet
# ============================================================================


class NamedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.get_name_path(path)

    @staticmethod
    def get_name_path(name):
        name = name.split('/')
        return osp.join(name[-2], name[-1])



class ImageFolderFiltered(NamedImageFolder):
    def __init__(
        self,
        filter_by=-1,
        label_query=-1,
        negate=False,
        partition='train',
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        predictions = pd.read_csv('utils/imagenet-{}-prediction-label-{}.csv'.format(partition, label_query))
        operator = (lambda x, y: x != y) if negate else (lambda x, y: x == y)

        data = predictions['idx'][operator(predictions['prediction'], filter_by)]


# ============================================================================
# Other dataset wrappers
# ============================================================================


class TextualDataset():
    def __init__(
        self,
        custom_tokens: list,
        base_prompt_generator,  # generates a varity of base prompts when called
        dataset
    ):
        '''
        Dataset to return an image + a generated prompt with the custom prompt.
            :custom_tokens: custom token(s) to train with
            :base_prompt_generator: function or object that returns the base prompt
                                    when using the custom_tokens as input
            :dataset: object dataset containing the images of interest to train the
                      token. When accessing to its datapoints (dataset[i]), it
                      should return only an image or the first item is the image.
        '''
        self.dataset = dataset
        self.custom_tokens = custom_tokens
        self.base_prompt_generator = base_prompt_generator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data[0] if isinstance(data, tuple) else data
        tokens = self.custom_tokens
        prompt = self.base_prompt_generator(tokens)
        return img, prompt


# ============================================================================
# Extracted from:
# https://github.com/guillaumejs2403/ACE/blob/main/guided_diffusion/sample_utils.py
# ============================================================================


class SlowSingleLabel():
    def __init__(self, label, dataset, maxlen=float('inf')):
        self.dataset = dataset
        self.indexes = []
        if isinstance(dataset, datasets.ImageFolder):
            self.indexes = np.where(np.array(dataset.targets) == label)[0]
            self.indexes = self.indexes[:maxlen]
        else:
            if label != -1:
                print('Slow route. This may take some time!')
                for idx, (_, l) in enumerate(tqdm(dataset)):

                    l = l['y'] if isinstance(l, dict) else l
                    if l == label:
                        self.indexes.append(idx)

                    if len(self.indexes) == maxlen:
                        break
            else:
                self.indexes = list(range(min(maxlen, len(dataset))))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]

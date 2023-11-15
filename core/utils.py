import os
import os.path as osp
import numpy as np
import itertools
import PIL.Image as Image

import torch


def Print(*args, **kwargs):
    print('_' * 50)
    print(*args, **kwargs)


def generate_prompt(args, target, pred, binary):
    '''
    :args: argparse object
    :target: target label, must be int
    :pred: prediction label, must be int
    :binary: is a binary dataset
    :pn: use positive/negative prompt 
    '''
    base = args.base_prompt
    if binary:
        # this is similar than the multi label class
        s = args.pos_custom_token if pred == 1 else args.neg_custom_token
        t = args.pos_custom_token if pred != 1 else args.neg_custom_token

        if args.dataset == 'CelebAHQ':
            sp = f'{base} with a {s}'
            tp = f'{base} with a {t}'

        elif args.dataset == 'BDD100k':
            sp = f'{base} indicating to {s}'
            tp = f'{base} indicating to {t}'

    else:
        sp = f'{base} with a ' + args.generic_custom_token.replace('&', str(pred))
        tp = f'{base} with a ' + args.generic_custom_token.replace('&', str(target))

    return sp, tp


def merge_chunks(args):
    # import here because we don't need it somewhere else
    import copy
    import yaml

    text_fn = lambda chunk: osp.join(args.output_path, 'Results', args.exp_name, f'c-{chunk}_{args.chunks}-summary.yaml')

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

    for chunk in range(0, args.chunks):

        with open(text_fn(chunk), 'r') as f:
            data = yaml.safe_load(f)

        for k, v in data['explanation'].items():
            stats['explanation'][k] += v * data['n']
        stats['explanation']['p'] *= data['explanation']['cf']

        for k, v in data.items():
            if k == 'explanation':
                continue
            if k == 'n':
                stats[k] += v
                continue
            stats[k] += data['n'] * v

    for k, v in stats['explanation'].items():
        stats['explanation'][k] /= stats['n']
    stats['explanation']['p'] /= stats['explanation']['cf']

    for k, v in stats.items():
        if k == 'explanation':
            continue
        if k == 'n':
            continue
        stats[k] /= stats['n']

    with open(osp.join(args.output_path, 'Results', args.exp_name, 'summary.yaml'), 'w') as f:
        f.write(str(stats))

    with open(osp.join(args.output_path, 'Results', args.exp_name, 'summary.yml'), 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)

    print(stats)


# =======================================================
# Extracted and modified from
# https://github.com/guillaumejs2403/ACE/blob/main/core/metrics.py
# =======================================================


@torch.no_grad()
def accuracy(logits, label, topk=(1, 5), binary=False):
    '''
    Computes the topx accuracy between the logits and label.
    If set the binary flag to true, it will compute the top1 and the rest will return 1.
    Additionally, it computes the probability of the label class.
    '''
    if binary:
        prob = torch.sigmoid(logits)
        prob = prob * (label == 1).float() + (1 - prob) * (label == 0).float()
        res = [((logits > 0).float() == label)]
        res += [torch.ones_like(res[0])] * (len(topk) - 1)
    else:
        maxk = max(topk)
        prob = torch.softmax(logits, dim=1)[range(label.size(0)), label]
        _, pred_k = torch.topk(logits, maxk, dim=1)
        correct_k = (pred_k == label.view(-1, 1))

        res = []
        for k in topk:
            res.append(correct_k[:, :k].sum(dim=1))

    return res, prob


@torch.no_grad()
def get_prediction(classifier, img, binary):
    log = classifier(img)
    if binary:
        pred = (log > 0).float()
    else:
        pred = log.argmax(dim=1)

    return log, pred


# =======================================================
# Extracted and modified from
# https://github.com/guillaumejs2403/ACE/blob/main/guided_diffusion/sample_utils.py
# =======================================================


class ImageSaver():
    def __init__(self, output_path, exp_name, extention='.png'):
        self.output_path = output_path
        self.exp_name = exp_name
        self.idx = 0
        self.extention = extention
        self.construct_directory()

    def construct_directory(self):

        os.makedirs(osp.join(self.output_path, 'Original', 'Correct'), exist_ok=True)
        os.makedirs(osp.join(self.output_path, 'Original', 'Incorrect'), exist_ok=True)

        for clst, cf, subf in itertools.product(['CC', 'IC'],
                                                ['CCF', 'ICF'],
                                                ['CF', 'Info', 'SM', 'Rec']):
            os.makedirs(osp.join(self.output_path, 'Results',
                                 self.exp_name, clst,
                                 cf, subf),
                        exist_ok=True)

    def __call__(self, imgs, cfs, recs, target, label,
                 pred, pred_cf, l_inf, l_1, indexes=None, masks=None):

        for idx in range(len(imgs)):
            current_idx = indexes[idx].item() if indexes is not None else idx + self.idx
            mask = None if masks is None else masks[idx]
            rec = None if recs is None else recs[idx]
            self.save_img(img=imgs[idx],
                          cf=cfs[idx],
                          rec=rec,
                          idx=current_idx,
                          target=target[idx].item(),
                          label=label[idx].item(),
                          pred=pred[idx].item(),
                          pred_cf=pred_cf[idx].item(),
                          l_inf=l_inf[idx].item(),
                          l_1=l_1[idx].item(),
                          mask=mask)

        self.idx += len(imgs)

    @staticmethod
    def select_folder(label, target, pred, pred_cf):
        folder = osp.join('CC' if label == pred else 'IC',
                          'CCF' if target == pred_cf else 'ICF')
        return folder

    @staticmethod
    def preprocess(img):
        '''
        remove last dimension if it is 1
        '''
        if img.shape[2] > 1:
            return img
        else:
            return np.squeeze(img, 2)

    def save_img(self, img, cf, rec, idx, target, label,
                 pred, pred_cf, l_inf, l_1, mask):
        folder = self.select_folder(label, target, pred, pred_cf)
        output_path = osp.join(self.output_path, 'Results',
                               self.exp_name, folder)
        img_name = f'{idx}'.zfill(7)
        orig_path = osp.join(self.output_path, 'Original',
                             'Correct' if label == pred else 'Incorrect',
                             img_name + self.extention)

        if mask is None:
            l0 = np.abs(img.astype('float') - cf.astype('float'))
            l0 = l0.sum(2, keepdims=True)
            l0 = 255 * l0 / l0.max()
            l0 = np.concatenate([l0] * img.shape[2], axis=2).astype('uint8')
            l0 = Image.fromarray(self.preprocess(l0))
            l0.save(osp.join(output_path, 'SM', img_name + self.extention))
        else:
            mask = mask.astype('uint8')
            mask = Image.fromarray(mask)
            mask.save(osp.join(output_path, 'SM', img_name + self.extention))

        img = Image.fromarray(self.preprocess(img))
        img.save(orig_path)

        cf = Image.fromarray(self.preprocess(cf))
        cf.save(osp.join(output_path, 'CF', img_name + self.extention))

        if rec is not None:
            rec = Image.fromarray(self.preprocess(rec))
            rec.save(osp.join(output_path, 'Rec', img_name + self.extention))


        to_write = (f'label: {label}' +
                    f'\npred: {pred}' +
                    f'\ntarget: {target}' +
                    f'\ncf pred: {pred_cf}' +
                    f'\nl_inf: {l_inf}' +
                    f'\nl_1: {l_1}')
        with open(osp.join(output_path, 'Info', img_name + '.txt'), 'w') as f:
            f.write(to_write)


def save_tokens_and_embeddings(
        sd_model,
        tokens: list,
        output: str,
    ):
    '''
    Save newly added tokens and their textual embeddings to avoid 
    storing n-times Stable Diffusion. The format will be:
                    dict['embs'] = torch.tensor
                    dict['words'] = list
                    where dict['words'][i] is related to the
                    dict['embs'][i] embedding 
        params:
            :sd_model: stable diffusion model from hugging face
            :tokens: list containing the added tokens
            :output: output file name
    '''

    token_embeds = sd_model.text_encoder.get_input_embeddings().weight.data.detach().cpu()
    placeholder_token_ids = sd_model.tokenizer.convert_tokens_to_ids(tokens)

    d = {
        'embs': token_embeds[placeholder_token_ids],
        'words': tokens
    }
    torch.save(d, output)


def load_tokens_and_embeddings(
        sd_model,
        files: list,
    ):
    '''
    Load textual embeddings-tokens files in n-times Stable Diffusion. 
    It loads more than one!
        params:
            :sd_model: stable diffusion model from hugging face
            :tokens: list containing the added tokens
    '''

    if len(files) == 0:
        print('No embedding files loaded')

    for file in files:
        # load file
        d = torch.load(file, map_location='cpu')

        # add tokens
        num_added_tokens = sd_model.tokenizer.add_tokens(d['words'])
        assert num_added_tokens == len(d['words']), f'There are some tokens already from file {file} in Stable Diffusion'
        sd_model.text_encoder.resize_token_embeddings(len(sd_model.tokenizer))

        # add embeddings
        token_embeds = sd_model.text_encoder.get_input_embeddings().weight.data
        placeholder_token_ids = sd_model.tokenizer.convert_tokens_to_ids(d['words'])
        token_embeds[placeholder_token_ids] = d['embs'].to(token_embeds.device, dtype=token_embeds.dtype)


def add_new_tokens(
        new_tokens: list,
        init_tokens: list,
        sd_model,
        dtype=torch.float32
    ):
    '''
    Initialized new textual embeddings-tokens in Stable Diffusion.
    It loads more than one!
        params:
            :new_tokens: new text-code to include
            :init_tokens: initialization tokens
            :sd_model: stable diffusion model from hugging face
            :dtype: torch type to load the tokens
    '''

    num_added_tokens = sd_model.tokenizer.add_tokens(new_tokens)
    assert num_added_tokens == len(new_tokens), 'No new token(s) added. Current model already has the proposed token(s)!'
    print('Added tokens:', num_added_tokens)
    
    sd_model.text_encoder.resize_token_embeddings(len(sd_model.tokenizer))
    token_embeds = sd_model.text_encoder.get_input_embeddings().weight.data
    
    placeholder_token_ids = sd_model.tokenizer.convert_tokens_to_ids(new_tokens)
    initializer_token_ids = sd_model.tokenizer.convert_tokens_to_ids(init_tokens)
    token_embeds[placeholder_token_ids] = token_embeds[initializer_token_ids]

    index_no_updates = torch.arange(len(sd_model.tokenizer)).view(-1, 1) == torch.tensor([placeholder_token_ids])
    index_no_updates = index_no_updates.sum(dim=1)
    index_no_updates = (index_no_updates == 0).to(token_embeds.device, dtype=dtype)

    return index_no_updates, placeholder_token_ids, initializer_token_ids

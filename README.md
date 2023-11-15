# Text-to-Image Models for Counterfactual Explanations: a Black-Box Approach

This is the official code for the WACV 2024 paper _Text-to-Image Models for Counterfactual Explanations: a Black-Box Approach_.


## Set up your working space

### Environment

Create and activate the conda environment.

```bash
conda env create -f time_env.yml
conda activate time
```

### Downloading the models and datasets

Please follow ACE's instructions [(link)](https://github.com/guillaumejs2403/ACE) to download the CelebA HQ and BDD100k classification models and the other models necessary for the evaluation.

To download the BDD100k datasets [here](https://bdd-data.berkeley.edu/). To prepare the CelebA HQ dataset, download it [here](https://github.com/switchablenorms/CelebAMask-HQ). Additionally, download the `list_eval_partition.txt` from this [link](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE?resourcekey=0-TD_RXHhlG6LPvwHReuw6IA) and modify it following [this comment](https://github.com/guillaumejs2403/DiME/issues/3#issuecomment-1721325695).

## Counterfactual Explanation Generation

### Training

Before generating counterfactual explanations with TIME, you first need to extract the predictions of the target classifier. To do so, first, you need to run the `get_predictions.py` python code. The resulting output is a `.csv` file stored in the utils folder, where one column is the image filename and the other its respective prediction.

Once completed, you need to train the context and class-specific textual embeddings. To do so, you need to use the `training.py` python code. We based our code on [this](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) jupyter notebook.

To train the context embedding, run the code as follows:
```bash
DATASET=name-your-dataset
PATHDATA=/path/to/data
CONTEXTTOKENS=context.pth  # output filename
LQ=0  # query label to train on, e.g. 0 for forward/stop binary task in bdd
CUSTOMTOKENS="'|<C*1>|' '|<C*2>|' '|<C*3>|'"
INITTOKENS="centered realistic celebrity"

python training.py \
    --output_path $CONTEXTTOKENS \
    --dataset $DATASET \
    --data_dir $PATHDATA \
    --label_query $LQ --training_label -1 \
    --custom_tokens $CUSTOMTOKENS \
    --custom_tokens_init $INITTOKENS \
    --phase context \
    --mini_batch_size 1 \
    --enable_xformers_memory_efficient_attention
```

Here, the SD model will learn the text embeddings linked with the `|<C*1>|`, `|<C*2>|`, and `|<C*3>|` text code. These embeddings will be warmed-up with the embeddings coupled with the words in `INITTOKENS`. The output is a small `.pth` file containing the token code and its learned text embedding.

To train the class-related bias tokens, run the same code but change the `--phase` flag to `class`, and the `--training-label` to 1 or 0:
```bash
DATASET=name-your-dataset
PATHDATA=/path/to/data
CONTEXTTOKENS=context.pth  # output filename
CLASSTOKEN0=class-0.pth
LQ=0
NEGTOKENS="'|<AN*1>|' '|<AN*2>|' '|<AN*3>|'"
INITTOKENS="serious serious serious"


# training tokens for binary task $LQ with prediction 1
python training.py \
    --embedding-files $CONTEXTTOKENS \
    --output_path $CLASSTOKEN0 \
    --dataset $DATASET \
    --data_dir $PATHDATA \
    --label_query $LQ --training_label 0 \
    --custom_tokens $NEGTOKENS \
    --custom_tokens_init $INITTOKENS \
    --phase class \
    --mini_batch_size 1 \
    --base_prompt 'A |<C*1>| |<C*2>| |<C*3>| photo' \
    --enable_xformers_memory_efficient_attention


CLASSTOKEN1=class-1.pth
POSTOKENS="'|<AP*1>|' '|<AP*2>|' '|<AP*3>|'"
INITTOKENS="smile smile smile"

# training tokens for binary task $LQ with prediction 0
python training.py \
    --embedding-files $CONTEXTTOKENS \
    --output_path $CLASSTOKEN1 \
    --dataset $DATASET \
    --data_dir $PATHDATA \
    --label_query $LQ --training_label 1 \
    --custom_tokens $POSTOKENS \
    --custom_tokens_init $INITTOKENS \
    --phase class \
    --mini_batch_size 1 \
    --base_prompt 'A |<C*1>| |<C*2>| |<C*3>| photo' \
    --enable_xformers_memory_efficient_attention
```


### Generation

Generating the explanations is straightforward. You need to use the `generate-ce.py` python file as follows:
```bash
CONTEXTTOKENS=context.pth
CLASSTOKEN0=class-0.pth
CLASSTOKEN1=class-1.pth
STEPS="15 20 25 35"
GS="4 4 4 4"
OUTPUTPATH=/path/to/results
LABEL_QUERY=31
LABEL_TARGET=-1
CLASSIFIERPATH=/path/to/classifier/weights

python generate-ce.py \
    --embedding_files $CONTEXTTOKENS $CLASSTOKEN0 $CLASSTOKEN1 \
    --use_negative_guidance_denoise \
    --use_negative_guidance_inverse \
    --guidance-scale-denoising $GS \
    --guidance-scale-invertion $GS \
    --num_inference_steps $STEPS \
    --output_path $OUTPUTPATH \
    --exp_name $EXPNAME \
    --label_target $LABEL_TARGET \
    --label_query $LABEL_QUERY \
    --neg_custom_token '|<AN*1>| |<AN*2>| |<AN*3>|' \
    --pos_custom_token '|<AP*1>| |<AP*2>| |<AP*3>|' \
    --base_prompt 'A |<C*1>| |<C*2>| |<C*3>| photo' \
    --chunks $CHUNKS --chunk $CHUNK \
    --enable_xformers_memory_efficient_attention \
    --partition 'val' --dataset $DATASET \
    --data_dir $PATHDATA \
    --classifier_path $CLASSIFIERPATH
```
The output system filenames are equal to the one in our previous paper [Adversarial Visual Counterfactual Explanations](https://github.com/guillaumejs2403/ACE). In this case, `STEPS` is the noise inversion level, and `GS` is the gradient scale.

### Evaluation

We evaluate our pipeline using [Adversarial Visual Counterfactual Explanations](https://github.com/guillaumejs2403/ACE) code.

## Citation

If you find our code or paper useful, please cite our work
```
@InProceedings{Jeanneret_2024_WACV,
      title     = {Text-to-Image Models for Counterfactual Explanations: a Black-Box Approach}, 
      author    = {Guillaume Jeanneret and Loïc Simon and Frédéric Jurie},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      month     = {January},
      year      = {2024}
}
```
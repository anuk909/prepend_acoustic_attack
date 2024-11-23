# Overview

This directory contains the final attack segments created for the project "Confusing Whisper: Universal Acoustic Adversarial Attacks on Speech Foundation Models." 
These segments exploit vulnerabilities in Whisper models by leveraging a learned acoustic realization of specific tokens (e.g., <fr> <startoflm> <ru>). When prepended to any speech signal, these segments effectively manipulate the model's behavior, made the infferce time longger or change the behivor of the model .

## Abstract
Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate _special tokens_ in their vocabulary, such as `<endoftext>`, to guide their language generation process.
was exploited by learning a universal acoustic realization of the token, which, when prepended to any speech signal, caused the model to ignore the input and transcribe only the token. This effectively "muted" the model, achieving a success rate of over 97% in silencing unseen speech inputs.
In our project, we extend this concept by exploring the vulnerabilities of other special tokens, particularly language tokens such as <ru> (Russian) and <fr> (French). These tokens are critical for guiding Whisper’s transcription process in multi-lingual settings. By targeting these tokens, we test the model's susceptibility to adversarial manipulation in diverse scenarios, including language-specific transcription and translation tasks. Additionally, we experimented with other tokens, such as <startoflm>, to examine how the attack behaves when targeting tokens associated with initiating tasks or guiding the model's behavior. This broader investigation provides deeper insights into Whisper's vulnerabilities and the transferability of adversarial attacks across a variety of tokens and tasks.

# Quick Start (Running the Code of innfernce )
We have uploaded all the pre-learnt universal acoustic adversarial attack segments in   `audio_attack_segments/arbitrary_tokens`. 
Open `evaluate_arbitrary_attack.ipynb ` and try evaluating them for yourself. observe how these attacks impact on the opreation of the modal. 
examples
## Package Installation

This code has been tested on python>=3.9.

Fork the repository and then git clone

`git clone https://github.com/<username>/prepend_acoustic_attack`


Install all necessary packages by creating a conda environment from the existing `environment.yml` file.

```
conda env create -f environment.yml
conda activate venv_gector
### Until we figure out better way to do it:
pip install -r extra_requirements.txt
```

# Learning a universal prepend acoustic attack# Attack Segments Documentation

to repdruce our results this opertation need to be done :
1. run the script `train_attack' with the proper  aruggemtns for attack_token 
2. run the `process.py` script with the correct parameters for `attack_model_path` and `save_path`.


## Generation Commands

### `attack_segment_fr.np.npy`
```bash
python train_attack.py \
    --model_name whisper-tiny-multi \
    --data_name librispeech \
    --attack_method audio-raw \
    --attack_command arbitrary \
    --attack_token 50265 \
    --max_epochs 40 \
    --clip_val 0.02 \
    --attack_size 10240 \
    --save_freq 10
```

### `attack_segment_ru.np.npy`
```bash
python train_attack.py \
    --model_name whisper-tiny-multi \
    --data_name librispeech \
    --attack_method audio-raw \
    --attack_command arbitrary \
    --attack_token 50263 \
    --max_epochs 40 \
    --clip_val 0.02 \
    --attack_size 10240 \
    --save_freq 10
```

### `attack_segment_a.np.npy`
```bash
python train_attack.py \
    --model_name whisper-tiny-multi \
    --data_name librispeech \
    --attack_method audio-raw \
    --attack_command arbitrary \
    --attack_token 64 \
    --max_epochs 40 \
    --clip_val 0.02 \
    --attack_size 10240 \
    --save_freq 10
```

### `attack_segment_slm.np.npy`
```bash
python train_attack.py \
    --model_name whisper-tiny-multi \
    --data_name librispeech \
    --attack_method audio-raw \
    --attack_command arbitrary \
    --attack_token 50360 \
    --max_epochs 40 \
    --clip_val 0.02 \
    --attack_size 10240 \
    --save_freq 10
```
The following subsections give example commands to run the training, evaluations and analysis necessary to reproduce the results in our paper.

## Standard Arguments for Attack Configuration
train_attack.py can be used to learn a universal acoustic attack on any of the Whisper models. The following extra arguments may be of use:

- `max_epochs` : Maximum number of epochs to run the gradient-descent based training to learn the universal attack. In the paper we have the following configurations: tiny (40), base (40), small (120) and medium (160).
- `bs` : The batch size for learning the attack
- `save_freq` : The frequency of saving the learnt attack audio segment during the learning of the attack.

You can also  see all the arguments used in the different scripts in `src/tools/args.py`.

The following arguments specify the attack configuration:

- `model_name` : The specific Whisper model to learn the universal attack on.
- `attack_method`: What form of acoustic attack to learn. For this paper, we always use `audio-raw`.
- `clip_val` : The maximum amplitude (for imperceptibility) of the attack audio segment. Set to `0.02` in the paper.
- `attack_size` : The number of audio frames in the adversarial audio segment. Standard setting is `10,240`, which is equivalent to 0.64 seconds of adversarial audio, for audio sampled at 16kHz.
- `data_name` : The dataset on which the universal attack is to be trained / evaluated. Note that training is on the validation split of the dataset.
- `task` : This can either be `transcribe` or `translate`. This specifies the task that the Whisper model is required to do. Note that `translate` is only possible for the multi-lingual models.
- `language`: The source audio language. By default is `en`.

In the paper, `tiny`, `base`, `small` and `medium` refer to the multi-lingual versions of the Whisper models, whilst `tiny.en`, `base.en`, `small.en` and `medium.en` refer to the English-only versions of the Whisper Models (this is the nomenclature used in the original Whisper paper). However, we use a slightly different naming convention in the codebase when specifying the model name as an argument to the scripts. The Table below gives the mapping from the names used in the paper and the equivalent names used in the codebase.

| Model name in paper | `model_name` in codebase |
| --------------- | ------------------- |
| tiny.en | whisper-tiny |
| tiny | whisper-tiny-multi |
| base.en | whisper-base |
| base | whisper-base-multi |
| small.en | whisper-small |
| small | whisper-small-multi |
| medium.en | whisper-medium |
| medium | whisper-medium-multi |

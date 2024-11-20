# Attack Segments Documentation

In this directory, we have the final attack segments created in the project "Confusing Whisper: Another Universal Acoustic Adversarial Attacks on Speech Foundation Models".

Each file was generated by `train_attack` and then processed using the `process.py` script with the correct parameters for `attack_model_path` and `save_path`.

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
import getpass
USERNAME = getpass.getuser()

LIBRISPEECH_DIR = f'/home/sharifm/teaching/tml-0368-4075/2024-spring/students/{USERNAME}/prepend_acoustic_attack/librispeech'
TEDLIUM_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/tedlium/tedlium/test/'
MGB_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/mvse/MGB-3/mgb3/test/'
ARTIE_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/artie-bias-corpus/data/'

def _librispeech(sub_dir):
    '''
        for clean audio, set `sub_dir' to dev_clean/test_clean as dev/test sets
        for noisy audio, set `sub_dir' to dev_other/test_other as dev/test sets
    '''
    return _process(f'{LIBRISPEECH_DIR}/{sub_dir}/audio_ref_pair_list', ['/home/rm2114/rds/rds-altaslp-8YSp2LXTlkY/data/', f'/home/sharifm/teaching/tml-0368-4075/2024-spring/students/{USERNAME}/prepend_acoustic_attack/'])
    

def _tedlium():
    '''
        Returns the test split for TedLium3 dataset
    '''
    return _process(f'{TEDLIUM_DIR}/audio_ref_pair_list')

def _mgb():
    '''
        Returns the test split for MGB-3 dataset
    '''
    return _process(f'{MGB_DIR}/audio_ref_pair_list', ['mq227', 'vr313'])

def _artie():
    '''
        Returns the test split for ARTIE BIAS dataset
    '''
    return _process(f'{ARTIE_DIR}/audio_ref_pair')


def _process(fname, replace_base_dir=None):
    audio_transcript_pair_list = []
    with open(fname, 'r') as fin:
        for line in fin:
            _, audio, ref = line.split(None, 2)
            ref = ref.rstrip('\n')

            if replace_base_dir is not None:
                # change audio path as per user
                audio = audio.replace(replace_base_dir[0], replace_base_dir[1])

            sample = {
                    'audio': audio,
                    'ref': ref
                }
            audio_transcript_pair_list.append(sample) 
    return audio_transcript_pair_list
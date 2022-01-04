import numpy as np
import pandas as pd
import glob
import os
import librosa
from tqdm import tqdm

from config import CONFIG

def process_file(filename):
    """
    Load a file and extract its features into a dataframe as a sequence of 
    overlapping spectrograms
    """
    # Get parameter data from config
    sr = CONFIG['preprocessing']['sr']
    keep_wave = CONFIG['preprocessing']['keep_wave']
    feature_type = CONFIG['preprocessing']['features']['type']
    n_fft = CONFIG['preprocessing']['features']['n_fft']
    win_length = CONFIG['preprocessing']['features']['win_length']
    hop_size = CONFIG['preprocessing']['features']['hop_size']
    n_bins = CONFIG['preprocessing']['features']['n_bins']
    chunk_size = CONFIG['preprocessing']['features']['chunk_size']
    overlap = CONFIG['preprocessing']['features']['overlap']
    padding = CONFIG['preprocessing']['features']['padding']

    # Compute derived parameters
    chunk_hop = chunk_size - overlap
    chunk_size_wav = chunk_size * hop_size
    overlap_wav = overlap * hop_size
    chunk_hop_wav = chunk_size_wav - overlap_wav

    df = dict()
    df['magnitude'] = []
    if keep_wave: df['wave'] = []
    try:
        # Load file
        audio, _ = librosa.load(filename, sr=sr)

        if feature_type != 'none':
            # Compute features
            if feature_type == 'stft':
                s = librosa.stft(audio, n_fft=n_fft-1, win_length=win_length, hop_length=hop_size)
            elif feature_type == 'cqt':
                s = librosa.cqt(audio, n_bins=n_bins, hop_length=hop_size)
            elif feature_type == 'mel':
                s = librosa.feature.melspectrogram(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_size, n_mels=n_bins)
            else:
                raise Exception()

            # compute number of chunks
            num_chunks = -((overlap - s.shape[1]) // chunk_hop)

            # pad data (or alternatively truncate data)
            if padding:
                s = np.pad(s, [(0, 0), (0, num_chunks * chunk_hop + overlap - s.shape[1])])
            else:
                s = s[:,:(num_chunks - 1) * chunk_hop + overlap]

            # Create overlapping trunks along new dimension
            s = np.array([s[:,((chunk_hop) * i):((chunk_hop) * i) + chunk_size] for i in range(num_chunks)])

        if keep_wave:
            # Same steps as above
            num_chunks_wav = -((overlap_wav - audio.shape[0]) // chunk_hop_wav)
            if padding:
                audio = np.pad(audio, (0, num_chunks_wav * chunk_hop_wav + overlap_wav - audio.shape[0]))
            else:
                audio = audio[:(num_chunks_wav - 1) * chunk_hop_wav + overlap_wav]
            audio = np.array([audio[((chunk_hop_wav) * i):((chunk_hop_wav) * i) + chunk_size_wav] for i in range(num_chunks_wav)])

        # Apply log to spectrograms
        if feature_type == 'stft' or feature_type == 'cqt': mag = librosa.amplitude_to_db(np.absolute(s))
        if feature_type == 'mel': mag = librosa.power_to_db(s, ref=np.max)

        # Add to dataframe
        if keep_wave: df['wave'] += [audio]
        if feature_type != 'none': df['magnitude'] += [mag]
    except Exception as e:
        raise e
    return pd.DataFrame(df)


def process_files(dbdir=CONFIG['preprocessing']['source'], file_cap=None):
    """
    iterates over audio files in a given directory and combines features into dataframe
    """
    file_types = CONFIG['preprocessing']['file_types']

    # Find all files with the configured audio extensions
    files = []
    for root, dirnames, filenames in os.walk(dbdir):
        for filename in filenames:
            for extension in file_types:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
    
    # Iterate over files
    dfs = []
    i = 0
    for f in tqdm(files, desc='Computing Features', smoothing=0.1):
        try:
            # Get features from file
            _df = process_file(f)

            # Add file path as a column
            _df['file'] = [os.path.relpath(f, dbdir)]

            dfs += [_df]
        except Exception as e:
            raise e
        i += 1
        if file_cap is not None and i >= file_cap: break
    # Return the concatenation of all the files
    return pd.concat(dfs, ignore_index=True)


def get_directory(path, pos=CONFIG['preprocessing']['category_level']):
    """
    get directory at specified position from a path
    """
    return os.path.normpath(path).split(os.path.sep)[pos]


def save_hdf(df, dest=CONFIG['preprocessing']['destination']):
    """
    save dataframe into an hdf5 file
    """
    category_level = CONFIG['preprocessing']['category_level']
    label_key = CONFIG['preprocessing']['hdf_label_key']

    # find the categories
    categories = df.apply(lambda x: get_directory(x['file'], category_level), axis=1).drop_duplicates().reset_index(drop=True)
    categories.to_hdf(dest, key=label_key)

    # Use a different key per label when saving to file
    for label in tqdm(categories, desc='Saving Dataset', smoothing=0.1):
        filtered_df = df[df.apply(lambda x: get_directory(x['file'], category_level), axis=1) == label]
        filtered_df.to_hdf(dest, key=label, mode='a')


if __name__ == "__main__":
    dbdir = CONFIG['preprocessing']['source']
    df = process_files(dbdir)

    dest = CONFIG['preprocessing']['destination']
    save_hdf(df, dest)
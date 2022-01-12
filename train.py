import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import CONFIG

from torch.nn.utils.rnn import pack_padded_sequence


def train(model, dataset):
    batch_size = CONFIG['training']['batch_size']
    shuffle = CONFIG['training']['shuffle']
    max_epochs = CONFIG['training']['max_epochs']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO dataset split

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_x_collate(device))

    model.to(device)
    
    training = True
    with tqdm(dataloader, unit='batch') as pbar:
        for epoch in range(max_epochs):
            for batch in pbar:
                print(batch[0].shape, batch[1].shape, batch[2].shape)
                packed_x = pack_padded_sequence(batch[0], batch[2], batch_first=True, enforce_sorted=False)
                print(packed_x.data.shape)

                training = False
                break
            if not training: break


def pad_x_collate(device):
    def _pad_x_collate(batch):
        (xx, yy) = zip(*batch)
        x_lens = torch.tensor([len(x) for x in xx]).to(device)

        xx = [torch.tensor(x)[:, None, :, :] for x in xx]
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).to(device)

        yy = torch.tensor(yy).to(device)

        return xx_pad, yy, x_lens
    return _pad_x_collate

if __name__ == "__main__":
    from dataset import H5SpecSeqDataset
    from model import CRNN_Classifier

    dataset = H5SpecSeqDataset()
    model = CRNN_Classifier()

    train(None, dataset)
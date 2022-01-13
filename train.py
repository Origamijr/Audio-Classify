import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.std import trange

from dataset import split_dataset
from utilities import MovingAverage, filter_kwargs

from config import CONFIG

def train(model: nn.Module, dataset: Dataset):
    """
    Sets up and performs the main training loop
    """
    # get training parameters from config file
    batch_size = CONFIG['training']['batch_size']
    eval_batch_size = CONFIG['training']['eval_batch_size']
    shuffle = CONFIG['training']['shuffle']
    max_epochs = CONFIG['training']['max_epochs']
    optimizer_params = CONFIG['training']['optimizer']
    logging = CONFIG['training']['enable_logging']
    log_dir = CONFIG['training']['log_dir']

    # enable cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # generate data split and corresponding dataloaders
    train_data, val_data, test_data = split_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_x_collate(device))
    val_loader = DataLoader(val_data, batch_size=eval_batch_size, collate_fn=pad_x_collate(device))
    test_loader = DataLoader(test_data, batch_size=eval_batch_size, collate_fn=pad_x_collate(device))

    # Setup optimizer    
    optimizer = optim.Adam(model.parameters(), **filter_kwargs(optimizer_params, adapt_f=optim.Adam))

    # Setup logging
    if logging: writer = SummaryWriter(log_dir)

    # main training loop
    with trange(max_epochs, unit='epoch') as pbar:
        for epoch in trange(max_epochs):
            # Train on training data
            train_loss, train_acc = train_on_data(model, train_loader, optimizer)

            # Evaluate on validation data
            val_loss, val_acc = evaluate_on_data(model, val_loader)

            # Log data
            if logging: 
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('acc/train', train_acc, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('acc/val', val_acc, epoch)
            pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)


def pad_x_collate(device):
    """
    Convert data to tensors and pad input sequences to be the length of the longest sequence.
    Corresponding sequence lengths are also returned to keep information on original length.
    """
    def _pad_x_collate(batch):
        (xx, yy) = zip(*batch)
        x_lens = torch.tensor([len(x) for x in xx]).to(device)

        xx = [torch.tensor(x)[:, None, :, :] for x in xx]
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).to(device)

        yy = torch.tensor(yy).to(device)

        return xx_pad, yy, x_lens
    return _pad_x_collate


def compute_accuracy(y_pred, y):
    """
    ^
    """
    pred_label = torch.argmax(y_pred, dim=1)
    return float((pred_label == y).float().sum() / len(y))


def train_on_data(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer):
    """
    Performs on epoch of training a model on a given dataset's dataloader.
    Returns the loss and accuracy of a model on a dataset's dataloader.
    """
    total_loss = MovingAverage()
    total_accuracy = MovingAverage()
    model.train()
    with tqdm(dataloader, desc='Training', unit='batch', leave=False) as pbar:
        for x, y, x_len in pbar:
            batch_size = x.shape[0]

            # Compute outputs
            y_pred, _ = model(x, x_len)
            loss = model.loss(y_pred, y)

            # Update model weights
            loss.backward()
            optimizer.step()

            # Compute metrics
            total_loss.add(loss.item(), count=batch_size)
            total_accuracy.add(compute_accuracy(y_pred, y), count=batch_size)
            pbar.set_postfix(loss=total_loss.value, acc=total_accuracy.value)
    return total_loss.value, total_accuracy.value



def evaluate_on_data(model: nn.Module, dataloader: DataLoader):
    """
    Returns the loss and accuracy of a model on a dataset's dataloader
    """
    total_loss = MovingAverage()
    total_accuracy = MovingAverage()
    with torch.no_grad():
        model.eval()
        with tqdm(dataloader, desc='Evaluating', unit='batch', leave=False) as pbar:
            for x, y, x_len in pbar:
                batch_size = x.shape[0]

                # Compute outputs
                y_pred, _ = model(x, x_len)
                loss = model.loss(y_pred, y)

                # Compute metrics
                total_loss.add(loss.item(), count=batch_size)
                total_accuracy.add(compute_accuracy(y_pred, y), count=batch_size)
                pbar.set_postfix(loss=total_loss.value, acc=total_accuracy.value)
    return total_loss.value, total_accuracy.value




if __name__ == "__main__":
    from dataset import H5SpecSeqDataset
    from model import CRNN_Classifier

    dataset = H5SpecSeqDataset()
    model = CRNN_Classifier()

    train(model, dataset)
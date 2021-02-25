import torch
import torchaudio

def get_label_set(train_set):
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    return labels

def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0,2,1)

def collate_fn(batch, labels):
    tensors, targets = [], []
    for waveform, _, label, *_  in batch:
        tensors += [waveform]
        targets += [label_to_index(label, labels)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def label_to_index(word, labels):
    return torch.tensor(labels.index(word))

def index_to_label(index, labels):
    return labels[index]

def transform(wav, sample_rate, new_sample_rate):
    trans = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    return trans(wav)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)


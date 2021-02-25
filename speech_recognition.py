################################################
# 원본 파일
###############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
from torch.utils.tensorboard import SummaryWriter

log_interval = 20
n_epoch = 1
model_save_path = './model'
test_model_path = './model/3model.pt'

writer = SummaryWriter('./log/experiment_1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# dataset setting
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__('./', download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return[os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == 'validation':
            self._walker = load_list('validation_list.txt')
        elif subset == 'testing':
            self._walker = load_list('testing_list.txt')
        elif subset == 'training':
            excludes = load_list('validation_list.txt') + load_list('testing_list.txt')
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


train_set = SubsetSC('training')
test_set = SubsetSC('testing')

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

print(f'Shape of waveform {waveform.size()}')
print(f'Sample rate of waveform : {sample_rate}')

# plt.plot(waveform.t().numpy())
# plt.show()

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
print(labels)

waveform_first, *_ = train_set[0]

waveform_second, *_ = train_set[1]

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)


def label_to_index(word):
    return torch.tensor(labels.index(word))

def index_to_label(index):
    return labels[index]

word_start = 'yes'
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(f'{word_start} --> {index} --> {word_recovered}')


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0,2,1)



def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_  in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

ten, tar = [], []
ten += [waveform_first]
ten += [waveform_second]
tar += [label_to_index(label)]
tar += [label_to_index(label)]

ten_p = pad_sequence(ten)
tar_s = torch.stack(tar)

batch_size = 256

if device == 'cuda':
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory
)

# get some random training audio
# dataiter = iter(train_loader)
# audio, labels = dataiter.next()



# sample data 들어보는 방법
writer.add_audio('audio_1', waveform_first, 0, sample_rate=sample_rate)
writer.add_audio('audio_1', waveform_second, 1, sample_rate=sample_rate)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2*n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2*n_channel, 2*n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2*n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2*n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)
print(model)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print(f'Number of parameters: {n}')

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

def train(model, epoch, log_interval):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        data = transform(data)
        output = model(data)

        loss = F.nll_loss(output.squeeze(), target)
        # 변화도를 0으로 만들고, 역전파 수행, 가중치 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        losses.append(loss.item())
    writer.add_scalar('training loss', running_loss/len(train_loader), epoch)
    print(f'###### training loss {running_loss/len(train_loader)} , {epoch}')
    # model save
    torch.save(model.state_dict(), os.path.join(model_save_path, 'ep_'+str(epoch)+'_model.pt'))


def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)

def test(model, epoch):
    model.eval()
    correct = 0
    running_loss = 0.0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        data = transform(data)
        output = model(data)

        loss = F.nll_loss(output.squeeze(), target)
        running_loss += loss.item()
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)



    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
    writer.add_scalar('testing loss', running_loss/len(test_loader), epoch)
    print(f'###### testing loss {running_loss/len(test_loader)} , {epoch}')



losses = []



for epoch in range(1, n_epoch+1):
    train(model, epoch, log_interval)
    test(model, epoch)
    scheduler.step()


def predict(tensor):
    tensor = tensor.to(device)
    tensor = transform(tensor)
    new_model = M5()
    new_model.to(device)
    new_model.load_state_dict(torch.load(test_model_path))
    tensor = new_model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor

waveform, sample_rate, utterance, *_ = train_set[-1]

print(f'Expected : {utterance}, Predicted : {predict(waveform)}')





writer.close()





















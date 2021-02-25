from dataset import *
from util import *
from model import *
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import pickle

def train(args):
    ## 트레이닝 파라메터 설정하기
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    log_interval = args.log_interval

    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir)
    train_set = SubsetSC('training')

    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    print(f'Shape of waveform {waveform.size()}')
    print(f'Sample rate of waveform : {sample_rate}')

    labels = get_label_set(train_set)
    # save labels as pickle file
    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

    new_sample_rate = 8000
    transformed = transform(waveform, sample_rate, new_sample_rate)


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
        collate_fn=partial(collate_fn, labels=labels),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    model = M5(n_input=transformed.shape[0], n_output=len(labels))
    model.to(device)
    print(model)

    n = count_parameters(model)
    print(f'Number of parameters: {n}')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    losses = []

    def train_f(model, epoch, log_interval):
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
        # model save
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ep_'+str(epoch)+'_model.pt'))


    def test(model, epoch):
        model.eval()
        correct = 0
        running_loss = 0.0

        test_set = SubsetSC('testing')
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(collate_fn, labels=labels),
            num_workers=num_workers,
            pin_memory=pin_memory
        )

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


    for epoch in range(1, num_epoch+1):
        train_f(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()



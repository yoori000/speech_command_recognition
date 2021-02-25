from dataset import *
from util import *
from model import *
import pickle

def predict_result(tensor, sample_rate, test_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    tensor = transform(tensor, sample_rate, 8000)
    new_model = M5()
    new_model.to(device)
    new_model.load_state_dict(torch.load(test_model_path))
    tensor = new_model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    with open('labels.pickle', 'rb') as f:
        labels = pickle.load(f)
    tensor = index_to_label(tensor.squeeze(), labels)
    return tensor

def predict(args):
    test_model_path = args.test_model_path
    test_set = SubsetSC('testing')
    waveform, sample_rate, utterance, *_ = test_set[-1]
    print(f'Expected : {utterance}, Predicted : {predict_result(waveform, sample_rate, test_model_path)}')


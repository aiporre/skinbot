
from skinbot.dataset import * 
from skinbot.config import read_config, Config
from skinbot.models import get_model
from skinbot.evaluations import plot_one_grad_cam
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
config = read_config()
C = Config()
C.set_config(config)

root = config['DATASET']['root']  # '# ./data'
labels = read_labels_xls(root, concat=True)


all_dataloader = get_dataloaders(config, batch=16, mode='test', target='cropSingle', fold_iteration=2)


## SHOW IMAGES

log_interval = 1
log_interval = 1
config = read_config()
root_dir = config["DATASET"]["root"]
best_or_last = 'best'
only_eval = True
fold = 0
model_name = 'resnet101'
fuzzy_labels = False
EPOCHS = 100
LR = 0.00001
display_info = True
# gpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_PATH = f"/media/{os.environ['USER']}/GG2/skin-project/models_4_last_CNN_models/best_models"


"""

           __..--''``---....___   _..._    __
 /// //_.-'    .-/";  `        ``<._  ``.''_ `. / // /
///_.-' _..--.'_    \                    `( ) ) // //
/ (_..-' // (< _     ;_..__               ; `' / ///
 / // // //  `-._,_)' // / ``--...____..-' /// / //
"""
def load_model_weights(model, models_path, fold):
    fold_dic = {}
    for f in os.listdir(models_path):
        _fold = int(f[f.index('fold=')+5])
        fold_dic[_fold] = f
    model_path  = fold_dic.get(fold, None)
    if model_path is not None:
        model.load_state_dict(torch.load(os.path.join(models_path,model_path)))
        print('model loaded: ', model_path)
    return model

def plot_sample(idx, predictions_csv):
    df = pd.read_csv(predictions_csv)
    # selects sample
    idx_fname = dataset.image_fnames.index(str(df.iloc[idx]['fnames']))
    sample_19 = dataset[idx_fname]
    x,y = sample_19
    x = torch.unsqueeze(x,0)
    # makes prediciont
    pred_19 = np.fromstring(df.iloc[idx]['y_prob'].replace('[','').replace(']',''), sep=',')
    # plot bars and image
    target_num_to_str = {v:k for k,v in C.labels.target_str_to_num.items()}
    fname_19 = os.path.join(dataset.images_dir, dataset.image_fnames[idx_fname])
    if False: # crop in target:
        image_19 = read_image(fname_19).numpy().transpose(1,2,0)
    else:
        image_19 = read_image(fname_19).numpy()
        cc = dataset._crop_boxes[dataset.image_fnames[idx_fname]]
        image_19 = crop_lesion(image_19, cc).transpose(1,2,0)
    pred = [float(f'{x:.3f}') for x in pred_19.squeeze().tolist()]
    plt.imshow(image_19)
    plt.title(os.path.basename(fname_19))
    plt.axis('off')
    plt.show()
    width = 0.3
    ind = np.array([0,1,2,3,4,5,6])
    if isinstance(y,int):
        y_19_true = np.zeros_like(ind)
        y_19_true[y] = 1
    else:
        y_19_true = y
    plt.bar(ind, y_19_true, width=width, label='y_true')
    plt.bar(ind+width, pred_19.squeeze(), width=width, label='y_pred')
    plt.axis('on')
    plt.title(os.path.basename(fname_19))
    plt.xticks(ind + width / 2, (target_num_to_str[v] for v in range(len(C.labels.target_str_to_num))), rotation=90)
    plt.legend()
    plt.show()


folds_num = 5

for _fold in range(folds_num):
    all_dataloader = get_dataloaders(config, batch=16, mode='test', target='cropSingle', fold_iteration=_fold)
    dataset = all_dataloader.dataset
    predictions_csv = os.path.join(os.getcwd(), f'predictions_fold={_fold}_resnet101_cropSingle.csv')
    for i in range(len(dataset)):
        print(f"fold = {_fold}, image name= {dataset.image_fnames[i]}, index={i}")
        plot_sample(i, predictions_csv)

# find the worst samples

folds_num = 5
df_all_preds = None
for _fold in range(folds_num):
    all_dataloader = get_dataloaders(config, batch=16, mode='test', target='cropSingle', fold_iteration=_fold)
    dataset = all_dataloader.dataset
    predictions_csv = os.path.join(os.getcwd(), f'predictions_fold={_fold}_resnet101_cropSingle.csv')
    if df_all_preds is None:
        df_all_preds = pd.read_csv(predictions_csv)
        df_all_preds['fold'] = [_fold]*len(dataset)
    else:
        df1 = pd.read_csv(predictions_csv)
        df1['fold'] = [_fold]*len(dataset)
        df_all_preds = pd.concat([df_all_preds,df1 ], ignore_index=True)


def compute_worst_errors(df, top=10):
    errors = (df['y_pred']-df['y_true']).apply(lambda x: float(x!=0))
    def str_to_numpy(x):
        return np.fromstring(x.replace(']','').replace('[',''), sep=',')
    how_much = errors * df['y_prob'].apply(str_to_numpy)
    errors_pred = errors * df['y_pred']
    print('all error #', sum(errors))
    how_much = np.array([x[i] for i, x in zip(errors_pred.astype(int), how_much)])
    df1 = df.copy()
    df1['how_much'] = how_much
    return df1, sum(errors)

all_errors, N_errors = compute_worst_errors(df_all_preds)

print('WORKST SAMPLES:>>>> ', N_errors)
print(compute_worst_errors.nlargest(56, 'how_much'))
all_errors.to_csv('all_errors_pred_classification.csv')



### gradints imporntance CAM

def plot_one_sample_fname(fname, df):
    # create dataset
    df_idx = df[df.fnames == fname].index[0]
    print(df_idx)
    print(df.iloc[df_idx])
    _fold = int(df.iloc[df_idx]['fold'])
    print('images fold: ', _fold)
    all_dataloader = get_dataloaders(config, batch=16, mode='test', target='cropSingle', fold_iteration=_fold)
    dataset = all_dataloader.dataset
    # selects sample
    for ifm in dataset.image_fnames:
        if ifm.startswith(fname.split(".")[0]):
            print(ifm)
    idx_fname = dataset.image_fnames.index(fname)
    sample_19 = dataset[idx_fname]
    x,y = sample_19
    x = torch.unsqueeze(x,0)
    # makes prediciont
    pred_19 = np.fromstring(df.iloc[df_idx]['y_prob'].replace('[','').replace(']',''), sep=',')
    # plot bars and image
    target_num_to_str = {v:k for k,v in C.labels.target_str_to_num.items()}
    fname_19 = os.path.join(dataset.images_dir, dataset.image_fnames[idx_fname])
    if False: # crop in target:
        image_19 = read_image(fname_19).numpy().transpose(1,2,0)
    else:
        image_19 = read_image(fname_19).numpy()
        cc = dataset._crop_boxes[dataset.image_fnames[idx_fname]]
        image_19 = crop_lesion(image_19, cc).transpose(1,2,0)
    pred = [float(f'{x:.3f}') for x in pred_19.squeeze().tolist()]
    plt.imshow(image_19)
    plt.title(os.path.basename(fname_19))
    plt.axis('off')
    plt.show()
    width = 0.3
    ind = np.array([0,1,2,3,4,5,6])
    if isinstance(y,int):
        y_19_true = np.zeros_like(ind)
        y_19_true[y] = 1
    else:
        y_19_true = y
    plt.bar(ind, y_19_true, width=width, label='y_true')
    plt.bar(ind+width, pred_19.squeeze(), width=width, label='y_pred')
    plt.axis('on')
    plt.title(os.path.basename(fname_19))
    plt.xticks(ind + width / 2, (target_num_to_str[v] for v in range(len(C.labels.target_str_to_num))), rotation=90)
    plt.legend()
    plt.show()
    model_name = "resnet101"
    model, optimizer = get_model(model_name, optimizer='SGD', lr=LR)
    model = load_model_weights(model, MODELS_PATH, _fold)
    model.eval()
    fig2, ax2 = plot_one_grad_cam(model, dataloader=all_dataloader, target_mode='cropSingle', index=idx_fname)
#     ax2.set_title(os.path.basename(fname_19))
    plt.show()

for filename in compute_worst_errors(df_all_preds).nlargest(56, 'how_much')['fnames']:
    plot_one_sample_fname(filename, df_all_preds)



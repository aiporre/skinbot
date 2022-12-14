import logging
import os

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from skinbot.dataset import get_dataloaders
from skinbot.config import read_config, Config
from skinbot.engine import create_classification_trainer, configure_engines, create_detection_trainer, \
    create_classification_evaluator, create_detection_evaluator
from skinbot.evaluations import prediction_all_samples, error_analysis
from skinbot.models import get_model
# from skinbot.transformers import num_classes, target_str_to_num



from skinbot.utils import validate_target_mode, configure_logging

C = Config()

def main(best_or_last='best',
         target_mode='single',
         model_name='resnet101',
         fold=0,
         epochs=10,
         batch_size=32,
         lr=0.001,
         momentum=0.8,
         optimizer='SGD',
         freeze='No',
         device='cuda',
         only_eval=False,
         patience=None,
         model_path=None,
         external_data=False,
         config_file='config.ini'):
    # log_interval = 1
    config = read_config(config_file)
    C = Config()
    C.set_config(config)
    configure_logging(config)
    # root_dir = config["DATASET"]["root"]
    # best_or_last = 'best'
    # only_eval = False
    # fold = 0
    # model_name = 'resnet101'
    # fuzzy_labels = True
    EPOCHS = epochs
    LR = lr  # 0.001
    display_info = True
    # target_mode = "single"
    # gpu device
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # prepare dataset
    assert validate_target_mode(target_mode, ['single', 'multiple', 'fuzzy'])
    _fold = fold if not external_data else None
    test_dataloader = get_dataloaders(config, batch=16, mode='test', fold_iteration=_fold, target=target_mode)
    train_dataloader = get_dataloaders(config, batch=16, mode='train', fold_iteration=_fold, target=target_mode)

    # prepare models
    model, optimizer = get_model(model_name, optimizer='SGD', lr=LR, momentum=momentum, freeze=freeze)
    # move model to gpu
    model.to(device)
    # create trainer and evaluator
    if 'detection' in target_mode:
        trainer = create_detection_trainer(model, optimizer, device=device)
        evaluator = create_detection_evaluator(model, device=device)
    else:
        trainer, criterion = create_classification_trainer(model, optimizer, target_mode, device=device)
        evaluator = create_classification_evaluator(model, criterion, target_mode, device=device)

    # configuration of the engines
    trainer, evaluator = configure_engines( model, optimizer, trainer, evaluator, train_dataloader,
                                           test_dataloader, config, display_info, fold, target_mode,
                                           model_name, best_or_last, patience, model_path)
    # ---------------------------
    # Run training
    # ---------------------------
    if not only_eval:
        trainer.run(train_dataloader, max_epochs=EPOCHS)
    else:
        logging.info('dataset statistics')
        all_dataloader = get_dataloaders(config, batch=16, mode='all')
        all_labels = []
        # collect all labels in a list
        if os.path.exists('./dataset_statistics.csv'):
            df_all = pd.read_csv('./dataset_statistics.csv')
        else:
            for x, y in all_dataloader:
                all_labels.extend(y.tolist())
            df_all = pd.DataFrame(all_labels, columns=['label'])
            target_num_to_str = {v: k for k, v in C.labels.target_str_to_num.items()}
            df_all['label_name'] = df_all['label'].apply(lambda x: target_num_to_str[x])
            # save the dataset statistics
            df_all.to_csv('./dataset_statistics.csv', index=False)
        # sns.set(style="darkgrid")
        ax = sns.countplot(x="label_name", data=df_all)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_ylabel('Number of instances')
        plt.show()
        logging.info('Running evaluations Train and test (in that order).')
        evaluator.run(train_dataloader)
        logging.info(f"TRAIN: evaluator.state.metrics {evaluator.state.metrics}")
        evaluator.run(test_dataloader)
        logging.info(f"TEST: evaluator.state.metrics' {evaluator.state.metrics} ")
        _fold = fold if not external_data else 'external'
        predictions_fname = f'./predictions_fold={_fold}_{model_name}_{target_mode}.csv'

        if os.path.exists(predictions_fname):
            df = pd.read_csv(predictions_fname)
        else:
            df = prediction_all_samples(model, test_dataloader, fold, target_mode)
            df.to_csv(predictions_fname, index=False)
        logging.info(df.head())
        logging.info('prediction_results.csv saved')
        df = error_analysis(df)
        logging.info(f"prediction summary: {df['error'].describe()}")
        class_names = list(C.labels.target_str_to_num.keys())
        report = classification_report(df['y_true'], df['y_pred'], labels=range(len(class_names)),
                                       target_names=class_names)
        logging.info(report)

        matrix = confusion_matrix(df['y_true'], df['y_pred'])
        accuracies = matrix.diagonal() / matrix.sum(axis=1)
        logging.info(' Accuracy per class:')
        for acc, class_name in zip(accuracies, class_names):
            logging.info(f'{class_name}: {acc}')

        logging.info('confusion matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=evaluator.state.metrics['cm'].numpy(),
                                      display_labels=class_names)
        disp.plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # main(target_mode='multiple', patience=None, epochs=100, fold=0)
    # main(target_mode='fuzzy', patience=15, epochs=100, fold=0)
    # main(target_mode='cropSingle', patience=15, epochs=100, fold=0)
    main(target_mode='single', patience=15, epochs=100, fold=0, config_file='config.ini', model_name='resnet50',)
    # main(target_mode='detectionSingle', model_name='faster_rcnn_resnet50_fpn', patience=15, epochs=100, fold=0)
    # main(target_mode='multiple', patience=15, epochs=100, fold=0)
    model_path = None
    # main(target_mode='multiple', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True)
    # main(target_mode='fuzzy', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True)
    # single training with split
    # main(target_mode='single', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True)
    # EXTERNAL DATA: single evaluation of external data
    # main(target_mode='single', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True, external_data=True)

    ## load model path
    # model = get_model("asd")
    # model_path = 'best_models/best_fold=0_resnet101_number_model_0_accuracy=0.9333.pt'
    # if model_path is not None:
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     logging.info('loaded model', model_path)

import logging
import os

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from skinbot.dataset import get_dataloaders 
from skinbot.config import read_config
from skinbot.evaluations import prediction_all_samples, error_analysis
from skinbot.losses import EuclideanLoss, CosineLoss, MulticlassLoss
from skinbot.models import get_model
from skinbot.transformers import num_classes, target_str_to_num

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, EarlyStopping


def get_last_checkpoint(path_models, fold, model_name, target_mode):
    prefix = f"last_fold={fold}_{model_name}_{target_mode}_checkpoint"
    iterations = [p.split('_')[-1].split('.')[0] for p in os.listdir(path_models) if p.endswith('.pt') and p.startswith(prefix)]
    iterations = [int(ii) for ii in iterations if ii.isnumeric()]
    if len(iterations) == 0:
        return None
    last_iteration = max(iterations)
    return f"{prefix}_{last_iteration}.pt"

def get_best_iteration(path_models, fold, model_name, target_mode):
    prefix = f"best_fold={fold}_{model_name}_{target_mode}_model"
    iterations = [p.split('=')[-1].split('.pt')[0] for p in os.listdir(path_models) if p.endswith('.pt') and p.startswith(prefix)]
    if len(iterations) == 0:
        return None
    iterations = [float(ii) for ii in iterations]
    last_iteration = max(iterations)
    last_iteration_end = f"={last_iteration:.4f}.pt"
    best_model_path = [p for p in os.listdir(path_models) if p.endswith(last_iteration_end) and p.startswith(prefix)][0]
    return best_model_path
def get_log_path(config):
    logger_dir = config['LOGGER']['logfilepath']
    logger_fname = config['LOGGER']['logfilename']
    return os.path.join(logger_dir, logger_fname)

def configure_logging(config):
    # configure the logging system
    log_level = config['LOGGER']['loglevel']
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    if config['LOGGER']['logtofile'] == 'True':
        logger_dir = config['LOGGER']['logfilepath']
        logger_fname = config['LOGGER']['logfilename']
        logger_path = os.path.join(logger_dir, logger_fname)
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        print('Logging to: ', logger_path, ' with level: ', log_level)
        logging.basicConfig(filename=logger_path, filemode='w', level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level)

def main(best_or_last='best',
         target_mode='single',
         model_name='resnet101',
         fold=0,
         epochs=10,
         batch_size=32,
         lr=0.001,
         momentum=0.8,
         optimizer='SGD',
         freeze=False,
         device='cuda',
         only_eval=False,
         patience=None,
         model_path=None,
         external_data=False,
         config_file = 'config.ini'):
    # log_interval = 1
    config = read_config(config_file)
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
    def validate_target_mode(_target_mode, comparable_items):
        if any([item in _target_mode.lower() for item in comparable_items]):
            return True
        else:
            return False
    assert validate_target_mode(target_mode, ['single', 'multiple', 'fuzzy'])
    _fold = fold if not external_data else None
    test_dataloader = get_dataloaders(config, batch=16, mode='test', fold_iteration=_fold, target=target_mode)
    train_dataloader = get_dataloaders(config, batch=16, mode='train', fold_iteration=_fold, target=target_mode)
    

    # prepare models
    model, optimizer = get_model(model_name, optimizer='SGD', lr=LR, momentum=momentum, freeze=freeze)
    # move model to gpu
    model.to(device)
    # make loss according to target mode
    if 'single' in target_mode.lower():
        criterion = torch.nn.CrossEntropyLoss()
    elif 'multiple' in target_mode.lower():
        criterion = MulticlassLoss()
    elif 'fuzzy' in target_mode.lower():
        criterion = CosineLoss()
    else:
        raise ValueError(f"target_mode={target_mode} is not supported")

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    def pred_in_prob(output):
        ''' convert prediction and target to one-hot vector '''
        y_pred, y = output
        y_pred_prob = torch.softmax(y_pred, dim=1)
        y_pred_class = torch.argmax(y_pred, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=num_classes)
        # TODO: fix evaluation broken
        if validate_target_mode(target_mode, ['fuzzy', 'multiple']):
            y_argmax = torch.argmax(y, dim=1)
        elif 'single' in target_mode.lower():
            y_argmax = y.long()
        else:
            raise ValueError(f"target_mode={target_mode} is not supported")
        y_onehot= torch.nn.functional.one_hot(y_argmax, num_classes=num_classes)
        return y_pred_onehot, y_onehot

    def pred_in_onehot(output):
        ''' convert prediction to one-hot vector  and taget into single label '''
        y_pred, y = output
        y_pred_prob = torch.softmax(y_pred, dim=1)
        y_pred_class = torch.argmax(y_pred, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=num_classes)

        if target_mode in ['fuzzy', 'multiple']:
            y_argmax = torch.argmax(y, dim=1)
        elif 'single' in target_mode.lower():
            y_argmax = y.long()
        else:
            raise ValueError(f"target_mode={target_mode} is not supported")
        return y_pred_onehot, y_argmax

    val_metrics = {
        "accuracy": Accuracy(output_transform=pred_in_prob, is_multilabel=True),
        "nll": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=num_classes, output_transform=pred_in_onehot),
        'cosine': Loss(CosineLoss()) if validate_target_mode( target_mode, ['fuzzy', 'multiple']) else Loss(torch.nn.CrossEntropyLoss()),
        'euclidean': Loss(EuclideanLoss()) if validate_target_mode( target_mode, ['fuzzy', 'multiple']) else Loss(torch.nn.CrossEntropyLoss()),
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    RunningAverage(output_transform=lambda x:x).attach(trainer, "loss")
    
    if display_info and torch.cuda.is_available():
        from ignite.contrib.metrics import GpuInfo
        GpuInfo().attach(trainer, name='gpu')
    if config['LOGGER']['logtofile'] == 'True':
        log_path = get_log_path(config)
        log_file_handler = open(log_path, 'w')
        pbar = ProgressBar(persist=True, file=log_file_handler)
    else:
        pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    # @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    # def log_training_loss(trainer):
        # print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_training_results(engine):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        #print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg accuracy: {avg_accuracy:.2f} "
            f"Avg loss: {avg_nll:.2f} "
            f"Avg Cosine: {metrics['cosine']:.2f}"
            f"Avg Euclidean: {metrics['euclidean']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_validation_results(engine):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        # print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} "
            f"Avg accuracy: {avg_accuracy:.2f} "
            f"Avg loss: {avg_nll:.2f} "
            f"Avg Cosine {metrics['cosine']:.2f}"
            f"Avg Euclidean: {metrics['euclidean']:.2f}")

        pbar.n = pbar.last_print_n = 0
    
    to_save = {"weights": model, "optimizer": optimizer}
    handler_ckpt = Checkpoint(
        to_save,
        save_handler=DiskSaver('models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f'last_fold={fold}_{model_name}_{target_mode}',
    )
    if best_or_last == 'last':
        last_checkpoint_path = get_last_checkpoint('models', fold, model_name, target_mode)
        if last_checkpoint_path is not None:
            last_checkpoint_path = os.path.join('models', last_checkpoint_path)
            to_load = to_save
            handler_ckpt.load_objects(to_load=to_load, checkpoint=last_checkpoint_path)
            logging.info(f'loaded last checkpoint {last_checkpoint_path}')

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), handler_ckpt)

    ## early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss
    if patience is not None:
        early_stop_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)

    to_save = {'model': model}
    handler_best = Checkpoint(
        to_save, 
        save_handler=DiskSaver('best_models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f'best_fold={fold}_{model_name}_{target_mode}',
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer)
    )

    if best_or_last == 'best':
        if model_path is not None:
            best_model_path = model_path
        else:
            best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)
        if best_model_path is not None:
            best_model_path = os.path.join('best_models', best_model_path)
            to_load = to_save
            handler_best.load_objects(to_load=to_load, checkpoint=best_model_path)
            logging.info(f'loaded best model {best_model_path}')

    evaluator.add_event_handler(Events.COMPLETED, handler_best)
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
            for x,y in all_dataloader:
                all_labels.extend(y.tolist())
            df_all = pd.DataFrame(all_labels, columns=['label'])
            target_num_to_str= {v:k for k,v in target_str_to_num.items()}
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
        logging.info(f"prediction summary: { df['error'].describe()}")
        class_names = list(target_str_to_num.keys())
        report = classification_report(df['y_true'], df['y_pred'], labels=range(len(class_names)), target_names=class_names)
        logging.info(report)

        matrix = confusion_matrix(df['y_true'], df['y_pred'])
        accuracies = matrix.diagonal()/matrix.sum(axis=1)
        logging.info(' Accuracy per class:')
        for acc, class_name in zip(accuracies, class_names):
            logging.info(f'{class_name}: {acc}')

        logging.info('confusion matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=evaluator.state.metrics['cm'].numpy(), display_labels=class_names)
        disp.plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    # main(target_mode='multiple', patience=None, epochs=100, fold=0)
    # main(target_mode='fuzzy', patience=15, epochs=100, fold=0)
    main(target_mode='cropSingle', patience=15, epochs=100, fold=0)
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


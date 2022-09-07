import os

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from skinbot.dataset import get_dataloaders 
from skinbot.config import read_config
from skinbot.evaluations import prediction_all_samples, error_analysis
from skinbot.models import get_model
from skinbot.transformers import num_classes, target_str_to_num

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver



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
class CosineLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, x, y):
        x = torch.softmax(x, dim=1)
        return torch.mean(1 - self.cosine(x, y))

def main():
    log_interval = 1
    log_interval = 1
    config = read_config()
    root_dir = config["DATASET"]["root"]
    best_or_last = 'last'
    only_eval = False
    fold = 0
    model_name = 'resnet101'
    fuzzy_labels = True
    EPOCHS = 100
    # prepare dataset
    if fuzzy_labels:
        target_mode = "fuzzy"
    else:
        target_mode = "number"

    test_dataloader = get_dataloaders(config, batch=16, mode='test', fold_iteration=fold, target=target_mode)
    train_dataloader = get_dataloaders(config, batch=16, mode='train', fold_iteration=fold, target=target_mode)
    

    # prepare models
    model, optimizer = get_model(model_name, optimizer='SGD')
    if fuzzy_labels:
        criterion = CosineLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion)

    def pred_in_prob(output):
        y_pred, y = output
        y_pred_prob = torch.softmax(y_pred, dim=1)
        y_pred_class = torch.argmax(y_pred_prob, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=num_classes)
        if fuzzy_labels:
            y_argmax = torch.argmax(y, dim=1)
        else:
            y_argmax = y
        y_onehot= torch.nn.functional.one_hot(y_argmax, num_classes=num_classes)
        return y_pred_onehot, y_onehot

    def pred_in_onehot(output):
        y_pred, y = output
        y_pred_prob = torch.softmax(y_pred, dim=1)
        y_pred_class = torch.argmax(y_pred_prob, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=num_classes)

        if fuzzy_labels:
            y_argmax = torch.argmax(y, dim=1)
        else:
            y_argmax = y.long()
        return y_pred_onehot, y_argmax

    val_metrics = {
        "accuracy": Accuracy(output_transform=pred_in_prob, is_multilabel=True),
        "nll": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=num_classes, output_transform=pred_in_onehot)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    RunningAverage(output_transform=lambda x:x).attach(trainer, "loss")
    
    display_info=False
    if display_info:
        from ignite.contrib.metrics import GpuInfo
        GpuInfo().attach(trainer, name='gpu')
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    # @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    # def log_training_loss(trainer):
        # print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        #print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        # print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )

        pbar.n = pbar.last_print_n = 0
    
    to_save = {"weights": model, "optimizer": optimizer}
    handler_ckpt = Checkpoint(
        to_save,
        save_handler=DiskSaver('./models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f'last_fold={fold}_{model_name}_{target_mode}',
    )
    if best_or_last == 'last':
        last_checkpoint_path = get_last_checkpoint('./models', fold, model_name, target_mode)
        if last_checkpoint_path is not None:
            last_checkpoint_path = os.path.join('./models', last_checkpoint_path)
            to_load = to_save
            handler_ckpt.load_objects(to_load=to_load, checkpoint=last_checkpoint_path)
            print('loaded last checkpoint', last_checkpoint_path)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), handler_ckpt)

    to_save = {'model': model}
    handler_best = Checkpoint(
        to_save, 
        save_handler=DiskSaver('./best_models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f'best_fold={fold}_{model_name}_{target_mode}',
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer)
    )

    if best_or_last == 'best':
        best_model_path = get_best_iteration('./best_models',fold, model_name, target_mode)
        if best_model_path is not None:
            best_model_path = os.path.join('./best_models', best_model_path)
            to_load = to_save
            handler_best.load_objects(to_load=to_load, checkpoint=best_model_path)
            print('loaded best model', best_model_path)

    evaluator.add_event_handler(Events.COMPLETED, handler_best)
    if not only_eval:
        trainer.run(train_dataloader, max_epochs=EPOCHS)
    else:
        print('dataset statistics')
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

        evaluator.run(test_dataloader)
        print('evaluator.state.metrics', evaluator.state.metrics)
        if os.path.exists('./predictions.csv'):
            df = pd.read_csv('./predictions.csv')
        else:
            df = prediction_all_samples(model, test_dataloader)
            df.to_csv('prediction_results.csv', index=False)
        print('prediction_results.csv saved')
        df = error_analysis(df)
        print('prediction summary: ', df['error'].describe())
        class_names = list(target_str_to_num.keys())
        report = classification_report(df['y_true'], df['y_pred'], target_names=class_names)
        print(report)

        matrix = confusion_matrix(df['y_true'], df['y_pred'])
        accuracies = matrix.diagonal()/matrix.sum(axis=1)
        print(' Accuracy per class:')
        for acc, class_name in zip(accuracies, class_names):
            print(f'{class_name}: {acc}')

        print('confusion matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=evaluator.state.metrics['cm'].numpy(), display_labels=class_names)
        disp.plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()
    


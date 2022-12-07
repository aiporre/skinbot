import copy
import logging
import math
import os

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, RunningAverage
from torch import distributed as dist

from skinbot.losses import MulticlassLoss, CosineLoss, EuclideanLoss
from skinbot.transformers import num_classes, target_weights
from skinbot.utils import validate_target_mode, get_log_path


def get_last_checkpoint(path_models, fold, model_name, target_mode, by_iteration=False):
    prefix = f"last_fold={fold}_{model_name}_{target_mode}_checkpoint"
    if by_iteration:
        iterations = [p.split('_')[-1].split('.')[0] for p in os.listdir(path_models) if p.endswith('.pt') and p.startswith(prefix)]
        iterations = [int(ii) for ii in iterations if ii.isnumeric()]
        if len(iterations) == 0:
            return None
        last_iteration = max(iterations)
        return f"{prefix}_{last_iteration}.pt"
    else:
        checkpoints = [p for p in os.listdir(path_models) if p.endswith('.pt') and p.startswith(prefix)]
        # sort checkpoints by creation time
        checkpoints = sorted(checkpoints, key=lambda x: os.path.getctime(os.path.join(path_models, x)))
        # return the last checkpoint created
        if len(checkpoints) == 0:
            return None
        return checkpoints[-1]

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

def prepare_batch(batch, device=None):
    x, y = batch
    x = list(xx.to(device, non_blocking=True) for xx in x)
    y = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in y]
    return x, y


def reduce_dict(loss_dict):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        names = []
        values = []
        for k, v in loss_dict.items():
            names.append(k)
            values.append(v)
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_losses = {k: v for k, v in zip(names, values)}
    return reduced_losses


def create_detection_trainer(model, optimizer, device=None):
    def update_model(engine, batch):
        model.train()

        x, y = copy.deepcopy(batch)
        x_process, y_process = prepare_batch(batch, device=device)

        loss_dict = model(x_process, y_process)
        loss_sum = sum(value for value in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # update the model weights
        engine.state.optimizer.zero_grad()
        # if the loss is infinite, the use have to know about it
        if not math.isfinite(loss_value):
            logging.info("Loss is {}, resetting loss and skipping training iteration".format(loss_value))
            logging.info('Loss values were: %f ' % loss_dict_reduced)
            logging.info(f"Input labels were:  {[yy['labels'] for yy in y]}")
            logging.info(f"Input boxes were: ', {[yy['boxes'] for yy in y]}")
            loss_dict_reduced = {k: torch.tensor(0) for k, v in loss_dict_reduced.items()}
        else:
            loss_sum.backward()
            engine.state.optimizer.step()

        if engine.state.warmup_scheduler is not None:
            engine.state.warmup_scheduler.step()

        return x, y, loss_dict_reduced
    engine = Engine(update_model)
    engine.state.optimizer = optimizer
    return engine

def create_detection_evaluator(model, device=None):
    def update_model(engine, batch):
        x, y = prepare_batch(batch, device=device)
        x_process = copy.deepcopy(x)

        torch.cuda.synchronize()
        with torch.no_grad():
            y_pred = model(x_process)

        y_pred = [{k: v.to(device) for k, v in t.items()} for t in y_pred]

        res = {yy["image_id"].item(): yy_pred for yy, yy_pred in zip(y, y_pred)}
        engine.state.coco_evaluator.update(res)

        x_process = y_pred = None

        return x, y, res

    return Engine(update_model)

def create_classification_trainer(model, optimizer, target_mode, device=None):
    # make loss according to target mode
    if 'single' in target_mode.lower():
        # compute inverse of class weights
        v_max = max(target_weights.values())
        target_values_norm = [v_max / v for v in target_weights.values()]
        target_weights_tensor = torch.tensor(target_values_norm, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=target_weights_tensor)
    elif 'multiple' in target_mode.lower():
        criterion = MulticlassLoss()
    elif 'fuzzy' in target_mode.lower():
        criterion = CosineLoss()
    else:
        raise ValueError(f"target_mode={target_mode} is not supported")

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    return trainer, criterion

def create_classification_evaluator(model, criterion, target_mode, device=None):

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
    return evaluator

def configure_engines(model,
                      optimizer,
                      trainer,
                      evaluator,
                      train_dataloader,
                      test_dataloader,
                      config,
                      display_info,
                      fold,
                      target_mode,
                      model_name,
                      best_or_last,
                      patience,
                      model_path):
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

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
        # print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
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
            logging.info(f'loaded best model {best_model_path}')
            handler_best.load_objects(to_load=to_load, checkpoint=best_model_path)
        else:
            logging.info(f'No best model found. starting from scratch')

    evaluator.add_event_handler(Events.COMPLETED, handler_best)
    return trainer, evaluator

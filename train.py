import os
import torch
from skinbot.dataset import get_dataloaders 
from skinbot.config import read_config
from skinbot.models import get_model
from skinbot.transformers import num_classes

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver



def get_last_checkpoint(path_models):
    iterations = [p.split('_')[-1].split('.')[0] for p in os.listdir(path_models) if p.endswith('.pt')]
    iterations = [int(ii) for ii in iterations if ii.isnumeric()]
    last_iteration = max(iterations)
    return f"checkpoint_{last_iteration}.pt"

def get_best_iteration(path_models):
    iterations = [p.split('=')[-1].split('.pt')[0] for p in os.listdir(path_models) if p.endswith('.pt')]
    iterations = [float(ii) for ii in iterations]
    last_iteration = max(iterations)
    last_iteration_end = f"={last_iteration:.4f}.pt"
    best_model_path = [p for p in os.listdir(path_models) if p.endswith(last_iteration_end)]
    return best_model_path[0]

def main():
    log_interval = 1
    log_interval = 1
    config = read_config()
    root_dir = config["DATASET"]["root"]
    # prepare dataset
    test_dataloader = get_dataloaders(config, batch=16, mode='test')
    train_dataloader = get_dataloaders(config, batch=16, mode='train')
    

    # prepare models
    model, optimizer = get_model('resnet101', optimizer='SGD')
    criterion = torch.nn.CrossEntropyLoss() 
    trainer = create_supervised_trainer(model, optimizer, criterion)

    def pred_in_prob(output):
        y_pred, y = output
        y_pred_prob = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred_class = torch.argmax(y_pred_prob, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=num_classes)
        y_onehot= torch.nn.functional.one_hot(y, num_classes=num_classes)
        return y_pred_onehot, y_onehot

    val_metrics = {
        "accuracy": Accuracy(output_transform=pred_in_prob, is_multilabel=True),
        "nll": Loss(criterion)
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
        to_save, save_handler=DiskSaver('./models', create_dir=True, require_empty=False), n_saved=2
    )
    last_checkpoint_path = get_last_checkpoint('./models')
    last_checkpoint_path = os.path.join('./models', last_checkpoint_path)
    to_load = to_save
    handler_ckpt.load_objects(to_load=to_load, checkpoint=last_checkpoint_path)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), handler_ckpt)

    to_save = {'model': model}
    handler_best = Checkpoint(
        to_save, 
        save_handler=DiskSaver('./best_models', create_dir=True, require_empty=False),
        n_saved=2, filename_prefix='best',
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer)
    )

    best_model_path = get_best_iteration('./best_models')
    evaluator.add_event_handler(Events.COMPLETED, handler_best)


    trainer.run(train_dataloader, max_epochs=5)



if __name__ == "__main__":
    main()
    


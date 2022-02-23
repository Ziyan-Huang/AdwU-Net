import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.network_training.AdaptiveUNetTrainer_search import AdaptiveUNetTrainer_search
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)

    
    args = parser.parse_args()

    network = args.network
    network_trainer = args.network_trainer
    task = args.task
    fold = args.fold
    plans_identifier = args.p

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)


    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")
 
    assert issubclass(trainer_class,
                        AdaptiveUNetTrainer_search), "network_trainer was found but is not derived from AdaptiveUNetTrainer_search"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=True,
                            deterministic=False,
                            fp16=True)

    validation_only = False
    trainer.initialize(not validation_only)

    if args.continue_training:
        trainer.load_latest_checkpoint()
    trainer.run_training()

if __name__ == "__main__":
    main()
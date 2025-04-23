import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        #assert isinstance(config["main"]["execute_steps"], ListConfig)
        #assert isinstance(config["main"]["execute_steps"], list), (f"Expected a list for main.execute_steps, got {type(config['main']['execute_steps']).__name__}: "f"{config['main']['execute_steps']}")
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        ## YOUR CODE HERE: call the preprocess step
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact":"raw_data.parquet:latest", # REMEMBER TO USE THE VERSION!!
                "artifact_name":"preprocessed_data.csv", # choosing this name as config["data"]["reference_dataset"] is called this
                "artifact_type":"preprocessed_data", # this is something i gave
                "artifact_description":"Data with processing applied"
            }
        )

    if "check_data" in steps_to_execute:

        ## YOUR CODE HERE: call the check_data step
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact":config["data"]["reference_dataset"],
                "sample_artifact":"preprocessed_data.csv:latest", #WHY?? THIS IS THE SAME AS REFERENCE DATASET
                "ks_alpha":config["data"]["ks_alpha"]
            }
        )

    if "segregate" in steps_to_execute:

        ## YOUR CODE HERE: call the segregate step
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                #"input_artifact":config["data"]["reference_dataset"],
                "input_artifact":"preprocessed_data.csv:latest",#(better to explicitly chain this to the output artifact of the "preprocess" component)
                "artifact_root":"data",
                "artifact_type":"segregated_data",#this is something i gave
                "test_size":config["data"]["test_size"],
                #"random_state":config["main"]["random_seed"],#or config["random_forest_pipeline"]["random_forest"]["random_state"]?
                #commenting out above, as required=False and default value provided in /segregate/run.py
                "stratify":config["data"]["stratify"]
            }
        )

    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        ## YOUR CODE HERE: call the random_forest step
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data":"data_train.csv:latest",
                "model_config":model_config,
                "export_artifact":config["random_forest_pipeline"]["export_artifact"],
                #if you don't want to export the inference artifact, set the above to "null".
                "random_seed":config["main"]["random_seed"],#or config["random_forest_pipeline"]["random_forest"]["random_state"]?
                "val_size":config["data"]["val_size"],# but the solution says config["data"]["test_size"]
                # what's the point of giving both config["data"]["val_size"] and config["data"]["test_size"]?
                # the "evaluate" component will anyway check on the entire data_test.csv artifact created earlier. 
                "stratify":config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

        ## YOUR CODE HERE: call the evaluate step
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                #"model_export":config["random_forest_pipeline"]["export_artifact"],
                #you cannot set latest tag above, so do it as follows:
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                # or like this: config['random_forest_pipeline']['export_artifact']+":latest"
                "test_data":"data_test.csv:latest"
            }
        )


if __name__ == "__main__":
    go()

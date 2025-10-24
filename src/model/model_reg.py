
# Register model
import json
import mlflow
import logging
from mlflow.tracking import MlflowClient
from requests.exceptions import ConnectionError


# Configure logging
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://ec2-34-224-86-44.compute-1.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Utility: Load experiment info
def load_model_info(file_path: str) -> dict:
    """Load model metadata from a JSON file."""
    try:
        with open(file_path, "r") as f:
            info = json.load(f)
        logger.debug(f"Loaded model info from {file_path}: {info}")
        return info
    except FileNotFoundError:
        logger.error(f"{file_path} not found.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


# Core: Model registration function
def register_model(model_name: str, model_info: dict):
    """Register the model in MLflow."""
    try:
        client = MlflowClient()
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model from URI: {model_uri}")

        # Create the registered model if it doesn't exist
        try:
            client.create_registered_model(model_name)
            logger.info(f"Created new registered model '{model_name}'.")
        except Exception:
            logger.debug(f"Model '{model_name}' already exists. Proceeding...") 

        # Register model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=model_info["run_id"],
            description="Auto-registered from pipeline"
        )
        logger.info(
            f"Model '{model_name}' registered successfully as version {model_version.version}"
        )

        # Assign alias 
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=model_version.version
            )
            logger.info(f"Alias 'champion' set to version {model_version.version}")
        except Exception as alias_error:
            logger.warning(f"Could not assign alias: {alias_error}")

        print(f"Model '{model_name}' registered successfully.")

    except ConnectionError:
        logger.error(f"Cannot reach MLflow server at {MLFLOW_TRACKING_URI}")
        print("Failed to connect to MLflow server. Check if it's running and accessible.")
        raise
    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        print(f"Error during model registration: {e}")
        raise


# Entry point
def main():
    try:
        model_info = load_model_info("experiment_info.json")
        model_name = "yt-chrome-plugin-model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error(f"Model registration process failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

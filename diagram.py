from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB
from diagrams.aws.analytics import Kinesis
from diagrams.aws.ml import Sagemaker
from diagrams.aws.integration import SQS
from diagrams.aws.storage import S3
from diagrams.aws.management import Cloudwatch

with Diagram("Email KPI Optimizer", show=True):
    with Cluster("Data Management Layer"):
        data_loader = EC2("Data Loader")
        data_preprocessor = EC2("Data Preprocessor")
        data_validator = EC2("Data Validator")
        constants_manager = S3("Constants & Mappings")
        model_versioning = S3("Model Versioning")

    with Cluster("Feature Engineering Layer"):
        open_rate_features = EC2("Open Rate Features")
        click_rate_features = EC2("Click Rate Features")
        opt_out_rate_features = EC2("Opt-out Rate Features")

    with Cluster("Model Management Layer"):
        model_training = Sagemaker("Model Training")
        model_evaluation = Sagemaker("Model Evaluation")
        model_selection = Sagemaker("Model Selection")
        model_versioning >> model_training >> model_evaluation >> model_selection

    with Cluster("User Interface Layer"):
        ui_main = EC2("Main Application")
        ui_components = EC2("UI Components")
        ui_dashboard = EC2("Interactive Dashboard")

    with Cluster("Integration Layer"):
        external_api = SQS("Groq API")
        api_handler = EC2("API Handler")

    with Cluster("Data Flow"):
        input_data = S3("Input Data")
        input_data >> data_loader >> data_preprocessor >> data_validator >> constants_manager
        constants_manager >> open_rate_features
        constants_manager >> click_rate_features
        constants_manager >> opt_out_rate_features

        open_rate_features >> model_training
        click_rate_features >> model_training
        opt_out_rate_features >> model_training

        model_selection >> ui_main
        model_selection >> ui_components
        model_selection >> ui_dashboard

        ui_main >> external_api
        external_api >> api_handler

        api_handler >> model_training
        api_handler >> model_evaluation
        api_handler >> model_selection

    Cloudwatch("Monitoring") >> ui_main
    Cloudwatch("Monitoring") >> ui_components
    Cloudwatch("Monitoring") >> ui_dashboard
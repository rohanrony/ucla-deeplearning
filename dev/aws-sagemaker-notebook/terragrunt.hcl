remote_state {
  backend = "s3"
  config  = {
    bucket         = dependency.s3.outputs.s3_bucket_name
    key            = "terraform/aws-sagemaker-notebook.tfstate"
    region         = dependency.s3.outputs.s3_bucket_region
    dynamodb_table = "ucla-deeplearning-terraform-lock"
  }
}

dependency "s3" {
  config_path = "../aws-s3"
}

dependency "sagemaker" {
  config_path = "../aws-sagemaker"
}

inputs = {
  sagemaker_config=dependency.sagemaker.outputs
}
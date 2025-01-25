# LLM Evaluation Framework

This repository contains the code for a framework that allows you to compare the performance of multiple Large Language Models (LLMs) on a dataset. The framework evaluates models using two metrics: ROUGE and BERT-Score. It also supports models hosted on Amazon Bedrock and those available on Hugging Face.

Features

Compare multiple models using ROUGE and BERT-Score metrics.

Visualize the performance of models.

Evaluate models from Hugging Face and Amazon Bedrock.

User-friendly interface using Streamlit.

Requirements

Python Libraries

Install the following Python libraries:

pip install streamlit pandas seaborn transformers rouge-score bert-score boto3 python-dotenv

Environment Variables

Create a .env file in the root directory and add the following keys for Amazon Bedrock integration:

AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_REGION=us-east-1

Replace <your_aws_access_key> and <your_aws_secret_key> with your AWS credentials.

Dataset Format

Your dataset should be a CSV file with the following columns:
 1- text 
 2- reference_summary

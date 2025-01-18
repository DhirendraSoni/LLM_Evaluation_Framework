LLM Comparison Framework

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

text: The input text for which a summary needs to be generated.

reference_summary: The reference (ground-truth) summary for the input text.

How to Run the Framework

Step 1: Clone the Repository

git clone <repository_url>
cd <repository_name>

Step 2: Run the Streamlit Application

Run the Streamlit app using the following command:

streamlit run <script_name>.py

Replace <script_name> with the name of the Python file (e.g., app.py).

Step 3: Upload Dataset

Open the application in your browser (the Streamlit link will appear in the terminal).

Upload your dataset (CSV file) with text and reference_summary columns.

Step 4: Enter Models

Enter the names of the models you want to compare. For example:

facebook/bart-large-cnn, google/pegasus-xsum, bedrock:amazon.titan-text-lite-v1

Models from Hugging Face: Provide the full model name (e.g., facebook/bart-large-cnn).

Models from Amazon Bedrock: Prefix the model name with bedrock: (e.g., bedrock:amazon.titan-text-lite-v1).

Step 5: Evaluate Models

Click the Evaluate Models button to compute ROUGE and BERT-Score metrics for all models.

Step 6: Visualize Results

The results will be displayed in a table, showing the average scores for:

ROUGE-1

ROUGE-2

ROUGE-L

BERT-Score

Example Workflow

Start the application:

streamlit run app.py

Upload a dataset (e.g., dataset.csv).

Enter models to compare (e.g., facebook/bart-large-cnn, google/pegasus-xsum).

Click Evaluate Models.

View the results table showing ROUGE and BERT-Score metrics for each model.

File Structure

.
├── app.py               # Main Streamlit application file
├── requirements.txt     # List of required Python libraries
├── .env                 # AWS credentials for Amazon Bedrock
├── dataset.csv          # Example dataset (optional)
├── README.md            # Documentation

Metrics Explained

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

Measures the overlap of n-grams between the predicted and reference summaries.

ROUGE-1, ROUGE-2, and ROUGE-L are computed.

BERT-Score:

Uses BERT embeddings to compute semantic similarity between the predicted and reference summaries.

Produces a more nuanced evaluation compared to ROUGE.

Troubleshooting

AWS Credentials Error:

Ensure your .env file is correctly configured with valid AWS credentials.

Make sure the AWS region matches the region where your Bedrock models are hosted.

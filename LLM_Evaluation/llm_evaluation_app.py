import streamlit as st
import os
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from boto3 import client, session
import json
import streamlit as st
from bert_score import score as bert_score


load_dotenv()

class LLMComparisonFramework:
    def __init__(self, models, dataset, max_summary_length=100):
        st.write("Initializing the framework with models and dataset.")
        print("Initializing the framework with models and dataset.")
        self.models = models  # List of model names
        self.dataset = dataset  # Dataset as a DataFrame with 'text' and 'reference_summary'
        self.max_summary_length = max_summary_length
        self.results = pd.DataFrame(columns=['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        self.bedrock_client = self.initialize_bedrock_client()

    def initialize_bedrock_client(self):
        print("Initializing AWS Bedrock client.")
        st.write("Initializing AWS Bedrock client.")
        try:
            aws_session = session.Session()
            client = aws_session.client(
                'bedrock-runtime',
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            print("AWS Bedrock client initialized successfully.")
            return client
        except Exception as e:
            print(f"Failed to initialize AWS Bedrock client: {e}")
            raise

    def load_model(self, model_name):
        st.write(f"Loading model: {model_name}")
        print(f"Loading model: {model_name}")
        if model_name.startswith("bedrock:"):
            return None, model_name  # Bedrock models do not use tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model

    def generate_summary(self, model, tokenizer, text):
        st.write("Generating summary for a given text.")
        print("Generating summary for a given text.")
        if isinstance(model, str) and model.startswith("bedrock:"):
            st.write("bedrock model id : ", model)
            return self.generate_bedrock_summary(model, text)

        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs, max_length=self.max_summary_length, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def client(self, service_name):
        st.write(f"Creating client for service: {service_name}")
        print(f"Creating client for service: {service_name}")
        import boto3
        return boto3.client(service_name)

    def generate_bedrock_summary(self, model_name, text):
        st.write(f"Generating summary using Amazon Bedrock model: {model_name}")
        print(f"Generating summary using Amazon Bedrock model: {model_name}")
        model_id = model_name.split(":")[1]  # Extract Bedrock model ID
        payload = {
            "inputText": text,
                    "textGenerationConfig": {
                        "maxTokenCount": 4096,
                        "stopSequences": [],
                        "temperature": 0,
                        "topP": 1
                    }
                    }
        try:
            response = self.bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
            st.write("Successfully generated summary using Bedrock.")
            print("Successfully generated summary using Bedrock.")
            return response["body"].read().decode("utf-8")
        except Exception as e:
            print(f"Error generating summary with Bedrock model {model_name}: {e}")
            raise

    def calculate_rouge(self, predicted, reference):
        st.write("Calculating ROUGE scores.")
        print("Calculating ROUGE scores.")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, predicted)
        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }
    # BERT-Score
    def calculate_bert_score(self, predicted, reference):
        st.write("Calculating BERT-Score.")
        print("Calculating BERT-Score.")
        P, R, F1 = bert_score([predicted], [reference], lang="en", verbose=True)
        return F1.mean().item()

    # Evaluate model
    def evaluate_model(self, model_name):
        st.write(f"Evaluating model: {model_name}")
        print(f"Evaluating model: {model_name}")
        tokenizer, model = self.load_model(model_name)
        rouge1_scores, rouge2_scores, rougeL_scores, bert_scores = [], [], [], []

        for _, row in self.dataset.iterrows():
            text = row['text']
            reference_summary = row['reference_summary']

            predicted_summary = self.generate_summary(model, tokenizer, text)
            rouge_scores = self.calculate_rouge(predicted_summary, reference_summary)
            bert_score_value = self.calculate_bert_score(predicted_summary, reference_summary)

            rouge1_scores.append(rouge_scores['ROUGE-1'])
            rouge2_scores.append(rouge_scores['ROUGE-2'])
            rougeL_scores.append(rouge_scores['ROUGE-L'])
            bert_scores.append(bert_score_value)

        new_result = {
            'Model': model_name,
            'ROUGE-1': sum(rouge1_scores) / len(rouge1_scores),
            'ROUGE-2': sum(rouge2_scores) / len(rouge2_scores),
            'ROUGE-L': sum(rougeL_scores) / len(rougeL_scores),
            'BERT-Score': sum(bert_scores) / len(bert_scores)
        }
        self.results = pd.concat([self.results, pd.DataFrame([new_result])], ignore_index=True)


    def evaluate_model_old(self, model_name):
        st.write(f"Evaluating model: {model_name}")
        print(f"Evaluating model: {model_name}")
        tokenizer, model = self.load_model(model_name)
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        for _, row in self.dataset.iterrows():
            text = row['text']
            reference_summary = row['reference_summary']

            predicted_summary = self.generate_summary(model, tokenizer, text)
            rouge_scores = self.calculate_rouge(predicted_summary, reference_summary)

            rouge1_scores.append(rouge_scores['ROUGE-1'])
            rouge2_scores.append(rouge_scores['ROUGE-2'])
            rougeL_scores.append(rouge_scores['ROUGE-L'])

        new_result = {
            'Model': model_name,
            'ROUGE-1': sum(rouge1_scores) / len(rouge1_scores),
            'ROUGE-2': sum(rouge2_scores) / len(rouge2_scores),
            'ROUGE-L': sum(rougeL_scores) / len(rougeL_scores)
        }
        self.results = pd.concat([self.results, pd.DataFrame([new_result])], ignore_index=True)

    def evaluate_all_models(self):
        st.write("Evaluating all models on the dataset.")
        print("Evaluating all models on the dataset.")
        for model_name in self.models:
            print(f"Evaluating {model_name}...")
            self.evaluate_model(model_name)

    def visualize_results(self):
        st.write("Visualizing ROUGE scores for all models.")
        #st.write("Visualization is not available as matplotlib is not installed.")
        st.write(self.results)

    def deploy_best_model(self):
        st.write("Selecting and deploying the best model based on ROUGE-L score.")
        print("Selecting and deploying the best model based on ROUGE-L score.")
        best_model_name = self.results.loc[self.results['ROUGE-L'].idxmax(), 'Model']
        print(f"Deploying best model: {best_model_name}")
        return self.load_model(best_model_name)

# Example Usage => Streamlit UI
if __name__ == "__main__":
    st.title("LLM Comparison Framework")

    st.write("Upload your dataset with 'text' and 'reference_summary' columns.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully.")
        st.write(dataset)

        st.write("Enter model names to compare, separated by commas.")
        model_input = st.text_input("Model names", "facebook/bart-large-cnn, google/pegasus-xsum, bedrock:amazon.titan-text-lite-v1")
        models = [model.strip() for model in model_input.split(",")]

        st.write("Initializing framework...")
        framework = LLMComparisonFramework(models, dataset)

        if st.button("Evaluate Models"):
            framework.evaluate_all_models()
            framework.visualize_results()

        # if st.button("Deploy Best Model"):
        #     framework.deploy_best_model()
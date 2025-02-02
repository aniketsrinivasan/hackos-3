from typing import List, Optional
import csv
from transformers import AutoModel, AutoTokenizer
import torch

from .model_class import Model, ModelPrediction, _error_types, _severities


class Benchmark:
    def __init__(self, model: Optional[Model] = None, dataset_path: Optional[str] = None,
                 delimiter: str = ","):
        self.model = model
        self.dataset_path = dataset_path
        self.delimiter = delimiter

        # Get metrics:
        self.metrics = self.model.get_prediction_metrics()

        # Load dataset as a list of ModelPrediction:
        self.dataset = self.load_dataset(dataset_path, delimiter=self.delimiter)

    @staticmethod
    def load_dataset(path: str, delimiter=",") -> List[ModelPrediction]:
        """
        Loads the dataset (from a CSV file) into a list of ModelPrediction.
        """
        _dicts = []
        with open(path, 'r') as csvfile:
            rows = csv.DictReader(csvfile, delimiter=delimiter)
            _dicts = [{k: v for k, v in row.items()} for row in rows]
        _dataset = [ModelPrediction.from_dict(_dict) for _dict in _dicts]
        return _dataset

    @staticmethod
    def load_dicts(path: str, delimiter=",") -> List[dict]:
        _dicts = []
        with open(path, 'r') as csvfile:
            rows = csv.DictReader(csvfile, delimiter=delimiter)
            _dicts = [{k: v for k, v in row.items()} for row in rows]
        return _dicts

    @staticmethod
    def get_similarity_loss(output: str, label: str) -> float:
        similarity_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        # Tokenize the input strings
        inputs1 = tokenizer(f"{output}", return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(f"{label}", return_tensors="pt", padding=True, truncation=True)

        # Generate embeddings for both strings
        with torch.no_grad():
            outputs1 = similarity_model(**inputs1)
            outputs2 = similarity_model(**inputs2)

        # Extract the embeddings (last hidden state of the first token [CLS])
        embedding1 = outputs1.last_hidden_state[:, 0, :]
        embedding2 = outputs2.last_hidden_state[:, 0, :]

        # Compute cosine similarity
        cosine_sim = torch.nn.CosineSimilarity(dim=1)
        similarity_score = cosine_sim(embedding1, embedding2).numpy()[0]
        similarity_loss = (1 - similarity_score) / 2
        return similarity_loss

    @staticmethod
    def _get_loss(pred, label, metric: str) -> float:
        _class_metrics = ["error_type", "severity"]
        if metric in _class_metrics:
            if metric == "error_type":
                if pred not in _error_types:
                    # Unrecognized pred, set to other:
                    pred = "other"
            elif metric == "severity":
                if pred not in _severities:
                    # Unrecognized pred, set to other:
                    pred = "other"
            loss = pred != label
            return loss
        # Otherwise, use semantic similarity function:
        loss = Benchmark.get_similarity_loss(pred, label)
        return loss

    def run_benchmark(self) -> tuple:
        """
        Runs the benchmark on the provided dataset.
        Returns a tuple of the losses (dict) and model predictions (list).
        """
        _prediction_metrics = self.model.get_prediction_metrics()
        losses = {_metric: 0 for _metric in _prediction_metrics}
        skipped_examples = 0

        # Keep track of the predictions:
        predictions = []

        for label in self.dataset:
            # Get prediction:
            label_dict = label.to_dict()
            # print(f"Running prediction on {label.input}")
            prediction = self.model.predict(label.input)

            if prediction is None:
                skipped_examples += 1
                continue

            predictions.append(prediction)

            # Calculate losses:
            prediction_dict = prediction.to_dict()
            for _metric in _prediction_metrics:
                _pred = prediction_dict[_metric]
                _label = label_dict[_metric]
                _loss = Benchmark._get_loss(_pred, _label, _metric)
                # Add loss:
                losses[_metric] += _loss

            # print current losses:
            print(f"Current losses: {losses}")

        data_length = len(predictions)
        return losses, predictions, data_length, skipped_examples

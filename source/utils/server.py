import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from source.models.adult import Adult
from source.models.compas import Compas
from datasets import DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.preprocessor import Divider
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

class ServerConfig:
    """Creates class for the Federated Learning network server."""
    def __init__(self, model: Adult | Compas, dataset: str, seed: int, target_feature: str):
        self.model = model
        self.dataset = dataset
        self.seed = seed
        self.target_feature = target_feature
        self.testloader = None
        self.sensitive_col_index = None
        if self.dataset == "adult":
            self.drop_columns = []
            self.partition_key = "train"
            self.dataset_url = "scikit-learn/adult-census-income"
            self.preprocessor = self.skew_and_split(
                sensitive_feature=self.model.sensitive_feature,
                sensitive_value=self.model.sensitive_value
                )
        else:
            self.drop_columns = ["age","race:African-American", "race:Asian", "race:Hispanic", "race:Native_American", "race:Other"]
            self.partition_key = "test"
            self.dataset_url = "imodels/compas-recidivism"
            self.preprocessor = None

    def skew_and_split(
            self, 
            sensitive_feature: str, 
            sensitive_value: str, 
            ):
        """Preprocessor function call for Federated Dataset"""
        def skew(dataset_dict: DatasetDict) -> DatasetDict:
            """Splits a DatasetDict into a training set and testing set. Returns the test set to the server."""
            split = Divider(
                divide_config={"train":0.8,"test":0.2},
                divide_split="train"
            )
            dataset_dict_split = split(dataset_dict)
            dataset_dict_split.pop("train")
            dataset = dataset_dict_split["test"]
            minority_dataset = dataset.filter(lambda x: x[sensitive_feature] == sensitive_value)
            majority_dataset = dataset.filter(lambda x: x[sensitive_feature] != sensitive_value)
            total_majority = len(majority_dataset)
            total_minority = len(minority_dataset)
            print(f"""Total Test data: {total_majority + total_minority}
            Minority = {total_minority} | {(total_minority / (total_minority+total_majority))*100}%
            Majority = {total_majority} | {(total_majority / (total_minority+total_majority))*100}%""")
            return dataset_dict_split
        return skew
    
    def load_centralized_dataset(self) -> None:
        """Load and split the centralized dataset. Tabular data is preprocessed for model. The testloader is then stored to self.testloader"""
        partitioner = DirichletPartitioner(
            num_partitions=self.model.num_partitions,
            partition_by=self.model.sensitive_feature,
            alpha=self.model.alpha,
            seed=self.seed
        )
        fds = FederatedDataset(
            dataset=self.dataset_url,
            partitioners={self.partition_key:partitioner},
            preprocessor=self.preprocessor,
            seed=self.seed
        )
        dataset = fds.load_split("test").with_format("pandas")[:]
        dataset.dropna(inplace=True)

        categorical_cols = dataset.select_dtypes(include=["object"]).columns
        ordinal_encoder = OrdinalEncoder()
        dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

        self.drop_columns.append(self.target_feature)
        X = dataset.drop(self.drop_columns, axis=1)
        y = dataset[self.target_feature]
        self.sensitive_col_index = X.columns.get_loc(self.model.sensitive_feature)
        numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features)]
        )
        X = preprocessor.fit_transform(X)

        X_test_tensor = torch.tensor(X, dtype=torch.float32)
        y_test_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.model.batch_size, shuffle=False)
        self.testloader = test_loader

    def group_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, x_sensitive_feature: np.ndarray)->dict[str:float]:
        """Returns fairness metrics from training. Metrics returned are demographic parity difference, equal opportunity distance, equalised odds difference, minority acc, majority acc."""
        # Gets the unique groups. As the data was originally a tensor, it works out the counts to determine the minority and majority groups.
        unique_groups, counts = np.unique(x_sensitive_feature, return_counts=True)
        if counts[0] > counts[1]:
            majority_group = unique_groups[0]
            minority_group = unique_groups[1]
        else:
            minority_group = unique_groups[0]
            majority_group = unique_groups[1]
        min_condition = (x_sensitive_feature == minority_group)
        maj_condition = (x_sensitive_feature == majority_group)
        x_sensitive_feature[min_condition] = 0.0
        x_sensitive_feature[maj_condition] = 1.0
        unique_groups, counts = np.unique(x_sensitive_feature, return_counts=True)
        metrics = {}
        for value in unique_groups:
            mask = (x_sensitive_feature == value)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # Get the predicted positive rate for Demographic Parity Diff
            ppr = np.mean(y_pred_group == 1)

            # Get the confusion matrix to find the Equal Opportunity and Equalised Odds Diff
            if np.sum(y_true_group == 1) > 0:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0.,1.]).ravel()
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)
            else:
                tpr = np.nan
                fpr = np.nan

            # Get accuracy of groups
            acc = accuracy_score(y_true_group, y_pred_group)
            metrics[value] = {'PPR': ppr, 'TPR': tpr, 'FPR': fpr, 'ACC': acc, 'CM':{'tn':int(tn), 'fp':int(fp), 'fn':int(fn), 'tp':int(tp)}}

        group_ids = list(metrics.keys())
        g1, g2 = group_ids[0], group_ids[1]
        print(f'=== Group Metrics ===')
        for key in metrics.keys():
            print(f"Group {key}: PPR = {metrics[key]['PPR']:.3f}, TPR = {metrics[key]['TPR']:.3f}, FPR = {metrics[key]['FPR']:.3f}, ACC = {metrics[key]['ACC']:.3f}")
        print("=== Fairness Violations ===")
        print(f"Demographic Parity: {abs(metrics[g1]['PPR']-metrics[g2]['PPR']):.3f}")
        print(f"Equalised Odds: {max(abs(metrics[g1]['TPR']-metrics[g2]['TPR']),abs(metrics[g1]['FPR']-metrics[g2]['FPR'])):.3f}")
        print(f"Equal Opportunity: {abs(metrics[g1]['TPR']-metrics[g2]['TPR']):.3f}")
        print(f"Equalised Accuracy: {abs(metrics[g1]['ACC']-metrics[g2]['ACC']):.3f}")
        demographic_parity = float(abs(metrics[g1]['PPR']-metrics[g2]['PPR']))
        equalised_odds = float(max(abs(metrics[g1]['TPR']-metrics[g2]['TPR']),abs(metrics[g1]['FPR']-metrics[g2]['FPR'])))
        equal_opportunity = float(abs(metrics[g1]['TPR']-metrics[g2]['TPR']))
        equalised_accuracy = float(abs(metrics[g1]['ACC']-metrics[g2]['ACC']))
        minority_acc = float(metrics[0.]['ACC'])
        majority_acc = float(metrics[1.]['ACC'])

        # Metrics are collected in a dict. The dict is returned in a MetricRecord to the server. A limitation of the MetricRecord object type is that it doesn't accept nested dicts.
        fairness_metrics = {
            "demographic_parity": demographic_parity, 
            "equalised_odds": equalised_odds, 
            "equal_opportunity": equal_opportunity,
            "equalised_accuracy": equalised_accuracy, 
            "min_accuracy": minority_acc, 
            "maj_accuracy": majority_acc,
            "min_tpr": metrics[0.]['TPR'],
            "min_fpr": metrics[0.]['FPR'],
            "min_tp": metrics[0.]['CM']['tp'],
            "min_fp": metrics[0.]['CM']['fp'],
            "min_tn": metrics[0.]['CM']['tn'],
            "min_fn": metrics[0.]['CM']['fn'],
            "maj_tpr": metrics[1.]['TPR'],
            "maj_fpr": metrics[1.]['FPR'],
            "maj_tp": metrics[1.]['CM']['tp'],
            "maj_fp": metrics[1.]['CM']['fp'],
            "maj_tn": metrics[1.]['CM']['tn'],
            "maj_fn": metrics[1.]['CM']['fn'],
            }
        return fairness_metrics

    def test(self, device: torch.device)->dict[str:float]:
        """Validate the model on the test set."""
        self.model.to(device)
        criterion = torch.nn.BCELoss()
        correct, loss = 0, 0.0
        total = 0
        y_true = []
        y_pred = []
        x_sensitive_feature = []
        with torch.no_grad():
            # y_batch is true
            # prediction is prediction
            # I can collect the fairness metrics by examining these in relationship to the sensitive attributes
            # Maybe collect all results into a list?
            for X_batch, y_batch in self.testloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = self.model(X_batch)
                batch_loss = criterion(outputs, y_batch)
                loss += batch_loss.item()
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                y_true.append(y_batch.flatten().float())
                y_pred.append(predicted.flatten().float())
                x_sensitive_feature.append(X_batch[:, self.sensitive_col_index].flatten())
        accuracy = correct / total
        loss = loss / len(self.testloader)
        y_true = torch.cat(y_true, dim=0).cpu()
        y_pred = torch.cat(y_pred, dim=0).cpu()
        x_sensitive_feature = torch.cat(x_sensitive_feature, dim=0).cpu()
        metrics = self.group_fairness_metrics(y_true.numpy(), y_pred.numpy(), x_sensitive_feature.numpy())
        metrics["eval_acc"] = accuracy
        metrics["eval_loss"] = loss
        return metrics
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
import logging
import random

# 设置随机种子
random_seed = random.randint(0, 10000)

# 设置日志文件名
log_file = f"chinese_bert_randomseed={random_seed}.log"

# 配置日志输出到文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# X_w2v_file = "/C24108/Wangsy/data2vec/hupu_subwords.npy"
X_w2v_file="/C24108/Wangsy/data2vec/hupu_data_jieba.npy"
data_path = "/C24108/Wangsy/data/hupu_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(X_w2v_file)
logger.info(data_path)

def load_labels():
    data = pd.read_csv(data_path)
    data = data.dropna(subset=["reply", "label"])
    return data["label"].tolist()


if os.path.exists(X_w2v_file):
    X_w2v = torch.tensor(np.load(X_w2v_file), dtype=torch.float32).to(device)
    labels = load_labels()
    all_labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
else:
    logger.error("BERT 转换后的向量文件或标签文件未找到。请检查路径是否正确。")
    raise FileNotFoundError("数据文件未找到")


def split_data_cv(X, labels, n_splits=10, random_state=None):
    if random_state is None:
        random_state = random_seed
    logger.info(f"Random seed for KFold: {random_state}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []

    X = X.cpu().numpy()
    labels = labels.cpu().numpy()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        cv_splits.append(
            (
                torch.tensor(X_train, dtype=torch.float32).to(device),
                torch.tensor(y_train, dtype=torch.long).to(device),
                torch.tensor(X_test, dtype=torch.float32).to(device),
                torch.tensor(y_test, dtype=torch.long).to(device),
                test_index,
            )
        )

    return cv_splits, random_state


cv_splits, random_seed = split_data_cv(X_w2v, all_labels_tensor)

# 定义 LSTM 模型
# class AdvancedLSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
#         super(AdvancedLSTMModel, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size,
#             hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#         )
#         self.fc1 = nn.Linear(hidden_size, 100)
#         self.fc2 = nn.Linear(100, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x.unsqueeze(1))  # 添加额外的维度
#         out = out[:, -1, :]  # 取最后一个时间步的输出
#         fc1_out = self.fc1(out)
#         output = self.fc2(fc1_out)
#         return output
# 定义 LSTM 模型·
class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # 添加 BatchNorm1d
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))  # 添加额外的维度
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.batch_norm(out)  # Batch Normalization
        fc1_out = self.fc1(out)
        output = self.fc2(fc1_out)
        return output


# 模型参数
input_size = X_w2v.shape[1]
hidden_size = 64
output_size = 2

# 定义损失函数
criterion = nn.CrossEntropyLoss()
lr=0.0001
# num_epochs = 10000
num_epochs=4000
test_accuracies, test_precisions, test_recalls = [], [], []
logger.info(f"lr= {lr} num_epochs = {num_epochs}")
for fold, (X_train, y_train, X_test, y_test, test_index) in enumerate(cv_splits):
    logger.info(f"Starting fold {fold + 1}/{len(cv_splits)}")

    model = AdvancedLSTMModel(
        input_size, hidden_size, output_size, num_layers=4, dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=1024,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test.to(device), y_test.to(device)),
        batch_size=1024,
        shuffle=False,
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}"
            )

    # Evaluate the model
    model.eval()
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average="macro")
    recall = recall_score(all_targets, all_predictions, average="macro")

    test_accuracies.append(accuracy)
    test_precisions.append(precision)
    test_recalls.append(recall)

    logger.info(
        f"Fold {fold + 1}: Accuracy={accuracy:.2%}, Precision={precision:.2%}, Recall={recall:.2%}"
    )

avg_accuracy = np.mean(test_accuracies) * 100
avg_precision = np.mean(test_precisions) * 100
avg_recall = np.mean(test_recalls) * 100
logger.info(f"Average Accuracy: {avg_accuracy:.2f}%")
logger.info(f"Average Precision: {avg_precision:.2f}%")
logger.info(f"Average Recall: {avg_recall:.2f}%")

results_df = pd.DataFrame(
    {
        "Fold": list(range(1, len(test_accuracies) + 1)),
        "Accuracy": test_accuracies,
        "Precision": test_precisions,
        "Recall": test_recalls,
        "Random Seed": random_seed,
    }
)

logger.info(
    f"Cross-validation results saved to 'chinese_bert_randomseed={random_seed}.log'."
)

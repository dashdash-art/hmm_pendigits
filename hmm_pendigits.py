import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === Load the data ===
def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

# === Set dataset paths ===
train_path = r'C:\Users\jason\Downloads\pen+based+recognition+of+handwritten+digits\pendigits.tra'
test_path = r'C:\Users\jason\Downloads\pen+based+recognition+of+handwritten+digits\pendigits.tes'

# === Load and scale data ===
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Reshape to sequences: (samples, 8 points, 2 coords) ===
X_train_seq = X_train.reshape(-1, 8, 2)
X_test_seq = X_test.reshape(-1, 8, 2)

# === Convert to motion vectors (dx, dy) ===
X_train_diff = np.diff(X_train_seq, axis=1)
X_test_diff = np.diff(X_test_seq, axis=1)

# === Train 1 HMM per digit ===
digit_hmms = {}
for digit in range(10):
    digit_data = X_train_diff[y_train == digit]
    lengths = [7] * len(digit_data)  # after diff, 7 points per sample

    model = hmm.GaussianHMM(n_components=15, covariance_type="diag", n_iter=100)
    model.fit(np.concatenate(digit_data), lengths)
    digit_hmms[digit] = model

# === Predict test set ===
y_pred = []
for seq in X_test_diff:
    scores = {digit: model.score(seq) for digit, model in digit_hmms.items()}
    predicted = max(scores, key=scores.get)
    y_pred.append(predicted)

# === Accuracy ===
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy * 100))

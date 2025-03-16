from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Custom Decision Tree Regressor (from ipynb file)
class DecisionTreeRegressorCustom:
    def __init__(self, min_samples_split=2):
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(y) < self.min_samples_split or len(set(y)) == 1:
            return np.mean(y)
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_idx], y[left_idx])
        right_subtree = self._build_tree(X[right_idx], y[right_idx])
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_variance = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                variance = np.var(y[left_idx]) * len(y[left_idx]) + np.var(y[right_idx]) * len(y[right_idx])
                if variance < best_variance:
                    best_feature, best_threshold, best_variance = feature, threshold, variance
        return best_feature, best_threshold

    def predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        if x[feature] <= threshold:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


# Function to load and preprocess data (similar to your ipynb code)
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    headers = data[0]
    return headers, data[1:]

def label_encode_column(column):
    unique_values = list(set(column))
    encoding = {val: idx for idx, val in enumerate(unique_values)}
    return [encoding[val] for val in column], encoding

def standardize_features(X):
    X = np.array(X, dtype=float)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds, means, stds


# Load the saved model and preprocessing objects (same as in ipynb)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Initialize Flask app
app = Flask(__name__, template_folder='.')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Convert categorical features using the label encoders
        for column, encoder in label_encoders.items():
            if column in data:
                data[column] = encoder[data[column]] if data[column] in encoder else -1
        
        # Convert features to numerical array
        features = np.array([float(data[key]) for key in data.keys()]).reshape(1, -1)
        
        # Standardize features using the previously saved scaler
        features_scaled = (features - scaler['means']) / scaler['stds']
        
        # Make prediction using the loaded model
        prediction = model.predict(features_scaled)

        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

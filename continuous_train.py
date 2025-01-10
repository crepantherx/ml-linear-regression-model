import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':

    # Initialize the model with incremental learning capability
    model = SGDRegressor(max_iter=1000, tol=1e-3)

    # Function to simulate new data
    def generate_data():
        while True:
            X = np.random.rand(10, 1) * 10  # Random input features
            y = 3 * X + 7 + np.random.randn(10, 1) * 0.5  # y = 3x + 7 + noise
            yield X, y

    # Continuous training loop
    data_stream = generate_data()
    for i in range(5):  # Simulate 5 iterations of new data batches
        X_batch, y_batch = next(data_stream)  # Get a new batch of data
        model.partial_fit(X_batch, y_batch.ravel())  # Incremental training

        # Evaluate the model on the same batch (or validation set)
        y_pred = model.predict(X_batch)
        mse = mean_squared_error(y_batch, y_pred)
        print(f"Iteration {i + 1}: MSE = {mse:.4f}")

    # Save the trained model
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training completed and saved to 'linear_model.pkl'.")
import numpy as np


def predict(X_p, X, theta):
    # Normalizing with respect to the original data
    nom = (X_p - X.min(axis=0)) * 2
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1

    # Multiplying with learned theta
    X_p_norm = -1 + nom / denom
    X_p_norm = np.hstack((np.ones((np.shape(X_p_norm)[0], 1)), X_p_norm))

    return np.matmul(X_p_norm, theta)


def compute_cost(X, y, theta):
    size = np.size(y)
    mean = 1 / (2 * size)
    h = np.matmul(X, theta)  # Hypothesis

    a = np.transpose(h - y)
    b = h - y
    J = mean * np.matmul(a, b)  # Cost

    return J


def gradient_decent(X, y, theta, alpha, iters):
    size = np.size(y)
    J_history = np.zeros((iters, 1))

    for i in range(0, iters):
        h = np.matmul(X, theta)  # Hypothesis
        DeltaJ = np.matmul((1 / size) * np.transpose(X), (h - y))
        theta = theta - alpha * DeltaJ  # Updating all theta simultaneously
        J_history[i] = compute_cost(X, y, theta)

    return [theta, J_history]


def feature_normalize(X):
    # Normalizes to values between -1 and 1
    nom = (X - X.min(axis=0)) * 2
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1

    return -1 + nom / denom


def main():
    csv = "winequality-red.csv"
    test = "test.csv"
    r = np.loadtxt(open(csv, "rb"), delimiter=",", skiprows=1)
    num_rows, num_cols = r.shape

    features = num_cols - 1
    examples = num_rows

    X = r[:, :features]
    y = r[:, features]
    theta = np.zeros(features + 1)

    X_norm = np.hstack((np.ones((examples, 1)), feature_normalize(X)))

    alpha = 0.015  # Step for gradient descent
    iters = 10000  # Iterations for gradient descent

    # J_history is used for plotting, which has not been implemented yet
    [theta, J_history] = gradient_decent(X_norm, y, theta, alpha, iters)
    np.savetxt("theta.csv", theta, delimiter=",")  # Save theta to avoid reruns of gradient descent

    p = np.loadtxt(open(test, "rb"), delimiter=",", skiprows=1)
    X_p = p[:, :]

    guesses = predict(X_p, X, theta)
    np.savetxt("guesses.csv", guesses, delimiter=",")  # Output of predictions


if __name__ == "__main__":
    main()

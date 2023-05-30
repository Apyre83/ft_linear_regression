import csv
import json
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000


def load_data(filename):
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    mileage = np.array([int(row["km"]) for row in data])
    price = np.array([int(row["price"]) for row in data])
    return mileage, price


def normalize_data(mileage, price):
    mileage_mean = np.mean(mileage)
    mileage_std = np.std(mileage)
    normalized_mileage = (mileage - mileage_mean) / mileage_std

    price_mean = np.mean(price)
    price_std = np.std(price)
    normalized_price = (price - price_mean) / price_std

    return normalized_mileage, normalized_price


def calcuate_theta(mileage, price):
    theta0 = 0
    theta1 = 0
    m = len(mileage)

    for _ in range(NUM_ITERATIONS):

        predictions = theta0 + theta1 * mileage
        tmp_theta0 = LEARNING_RATE * (1/m) * np.sum(predictions - price)
        tmp_theta1 = LEARNING_RATE * (1/m) * np.sum((predictions - price) *
                                                    mileage)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1


def main():
    mileage, price = load_data("data.csv")
    mileage, price = normalize_data(mileage, price)
    theta_0, theta_1 = calcuate_theta(mileage, price)
    with open("theta.json", "w") as f:
        print("theta_0:", theta_0)
        print("theta_1:", theta_1)
        json.dump({"theta0": theta_0, "theta1": theta_1}, f, indent=4)

    plt.scatter(mileage, price, color='blue', label='Data Points')
    predicted_price = [theta_0 + theta_1 * m for m in mileage]
    plt.plot(mileage, predicted_price, color='red', label='Regression Line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


main()

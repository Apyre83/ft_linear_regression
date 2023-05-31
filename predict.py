import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.tracebacklimit = 0


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


def main():

    try:
        new_mileage = int(input("Please enter the mileage: "))
    except ValueError:
        print("Error: mileage must be an integer")
        sys.exit(1)

    try:
        with open("theta.json", "r") as f:
            theta = json.load(f)
            theta0 = theta["theta0"]
            theta1 = theta["theta1"]
    except FileNotFoundError:
        theta0 = 0
        theta1 = 0

    mileage, price = load_data("data.csv")

    mileage_mean = np.mean(mileage)
    mileage_std = np.std(mileage)
    price_mean = np.mean(price)
    price_std = np.std(price)

    mileage, price = normalize_data(mileage, price)

    new_mileage_normalized = (new_mileage - mileage_mean) / mileage_std
    predicted_price_normalized = theta0 + theta1 * new_mileage_normalized
    predicted_price = (predicted_price_normalized * price_std) + price_mean
    print("Predicted Price:", predicted_price)

    plt.scatter(mileage, price, color='blue', label='Data Points')
    predicted_price = [theta0 + theta1 * m for m in mileage]
    plt.plot(mileage, predicted_price, color='red', label='Regression Line')
    plt.plot(new_mileage_normalized, predicted_price_normalized, 'go')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


main()

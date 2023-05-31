import sys
import json
import csv
import numpy as np

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

    with open("theta.json", "r") as f:
        theta = json.load(f)
        theta0 = theta["theta0"]
        theta1 = theta["theta1"]

    mileage, price = load_data("data.csv")

    mileage_mean = np.mean(mileage)
    mileage_std = np.std(mileage)
    price_mean = np.mean(price)
    price_std = np.std(price)

    normalized_mileage, normalized_price = normalize_data(mileage, price)

    predicted_prices_normalized = theta0 + theta1 * normalized_mileage
    predicted_prices = (predicted_prices_normalized * price_std) + price_mean
    errors = np.abs(predicted_prices - price)


    precision = np.mean(100 * (1 - errors / 5000))
    print(f"La precision est de {precision}%")


main()

import pandas as pd

train = pd.read_csv(
    "../../data/1 训练用/Training set.csv", usecols=list(range(1, 15)) + [28]
)
valid = pd.read_csv(
    "../../data/2 测试用/Online Test set.csv", usecols=list(range(1, 15)) + [28]
)
test = pd.read_csv(
    "../../data/submission_sample.csv", usecols=list(range(1, 15)) + [28]
)


data = pd.concat([valid, test], axis=0)
data.sort_values(["Start"], ascending=True, inplace=True)
data["delta"] = pd.to_datetime(data["End"]) - pd.to_datetime(data["Start"])
print(data[["Start", "delta", "ProcessingResultIsOK"]])

# print(valid['Start'])

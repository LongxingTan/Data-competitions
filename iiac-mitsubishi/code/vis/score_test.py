from sklearn.metrics import accuracy_score, f1_score, recall_score

true = [1] + [0] * 13
true = [1, 1] + [0] * 12

# pred = [0] * 14
pred = [0, 1] + [0] * 12
# pred = [1, 1] + [0] * 12

print(accuracy_score(true, pred))
print(recall_score(true, pred))
print(f1_score(true, pred))

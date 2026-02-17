import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

data = pd.DataFrame({
    'hours_studied': np.random.randint(1, 10, 100),
    'attendance': np.random.randint(50, 100, 100),
    'quiz_score': np.random.randint(30, 100, 100)
})

data['result'] = (
    (data['hours_studied']*2 + data['attendance']*0.3 + data['quiz_score']*0.5) > 80
).astype(int)

X = data[['hours_studied','attendance','quiz_score']]
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

plt.scatter(data['quiz_score'], data['result'])
plt.xlabel("Quiz Score")
plt.ylabel("Pass/Fail")
plt.title("Student Analytics")
plt.show()

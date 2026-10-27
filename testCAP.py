import numpy as np
from signall import generate_cap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Generate CLEAN (noiseless) signals
cap16 = [generate_cap(16) for _ in range(500)]
cap64 = [generate_cap(64) for _ in range(500)]

X = np.array(cap16 + cap64)
y = np.array([0]*500 + [1]*500) # 0 for CAP-16, 1 for CAP-64

# 2. Split and train a simple classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print(f"Clean Signal Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")
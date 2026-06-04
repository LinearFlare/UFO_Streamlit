import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载并清洗数据
ufos = pd.read_csv("./data/ufos.csv")
print(f"Raw data shape: {ufos.shape}")

ufos = pd.DataFrame({
    "Seconds": ufos["duration (seconds)"],
    "Country": ufos["country"],
    "Latitude": ufos["latitude"],
    "Longitude": ufos["longitude"],
})

ufos.dropna(inplace=True)
ufos = ufos[(ufos["Seconds"] >= 1) & (ufos["Seconds"] <= 60)]
print(f"Cleaned data shape: {ufos.shape}")

# 2. 标签编码 (确保顺序: Australia, Canada, Germany, UK, US)
le = LabelEncoder()
ufos["Country"] = le.fit_transform(ufos["Country"])
print(f"Country mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 🌟 将 X 转换为纯 NumPy 数组进行训练，彻底解决线上预测时“特征名字不匹配”的警告
X = ufos[["Seconds", "Latitude", "Longitude"]].values
y = ufos["Country"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 🌟【已修复】改用 'lbfgs' 求解器，完美支持多分类任务，不再有本地运行报错
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(X_train, y_train)

# 🌟【核心兼容补丁】手动塞入此属性。不管线上 Render 用的 sklearn 是老版本还是新版本，都绝对不会再报 AttributeError 崩溃！
model.multi_class = "ovr"

# 3. 评估模型
predictions = model.predict(X_test)
print(f"\n{classification_report(y_test, predictions)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

# 4. 创建文件夹并保存模型
os.makedirs("model", exist_ok=True)
model_path = "model/ufo-model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved successfully to {model_path}")

# 5. 测试验证
test_data = [[50.0, 44.0, -12.0]]
test_pred = model.predict(test_data)
countries = ["Australia", "Canada", "Germany", "UK", "US"]
print(f"Test prediction for [50s, lat=44, lon=-12]: {countries[test_pred[0]]}")
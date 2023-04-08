import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.impute import SimpleImputer
from my_functions import evaluate_model
from sklearn.feature_selection import mutual_info_classif
import os

result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 1. 加载数据集
file_path = 'DATA/application_data.csv'
df = pd.read_csv(file_path)

# 2. 检查数据集是否成功加载
print(df.head())

# 3. 进行特征工程和特征选择
# 3.1 特征工程
df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

# 3.2 特征编码
le = LabelEncoder()
df = pd.get_dummies(df, columns=['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                                  'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                                  'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
                                  'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',
                                  'EMERGENCYSTATE_MODE'])

df['CODE_GENDER'] = le.fit_transform(df['CODE_GENDER'])
df['FLAG_OWN_CAR'] = le.fit_transform(df['FLAG_OWN_CAR'])
df['FLAG_OWN_REALTY'] = le.fit_transform(df['FLAG_OWN_REALTY'])

# 3.3 处理缺失值
imputer = SimpleImputer(strategy='median')
df_imputed = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)

# 3.4 特征选择
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']
selector = SelectKBest(mutual_info_classif, k=20)
selector.fit(X, y)

# 保存选择的特征名称
selected_features = pd.DataFrame({'feature': X.columns[selector.get_support()]})
selected_features.to_csv(os.path.join(result_dir, 'selected_features.csv'), index=False)

# 应用选择器
X_selected = selector.transform(X)

# 3.5 可视化和保存特征选择结果
selected_features = pd.DataFrame({'feature': X.columns[selector.get_support()],
                                  'importance': selector.scores_[selector.get_support()]})
selected_features.sort_values(by='importance', ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
plt.bar(selected_features['feature'], selected_features['importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 20 Important Features')
plt.savefig(os.path.join(result_dir, 'top_20_features.png'), bbox_inches='tight')
plt.show()

# 4. 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 5. 训练和评估模型
# 5.1 随机森林模型
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# 5.2 XGBoost模型
clf_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)
clf_xgb.fit(X_train, y_train)

# 6. 模型评估和可视化
selected_feature_names = X.columns[selector.get_support()]
evaluate_model(clf_rf, X_test, y_test, model_name='RandomForest', selected_features=selected_feature_names)
evaluate_model(clf_xgb, X_test, y_test, model_name='XGBoost', selected_features=selected_feature_names)
plt.show()

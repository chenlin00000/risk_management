import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.impute import SimpleImputer
from my_functions import evaluate_model
from sklearn.feature_selection import mutual_info_classif
import os
import seaborn as sns

result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 1. 加载数据集
file_path = 'DATA/application_data.csv'
df = pd.read_csv(file_path)

# 2. 检查数据集是否成功加载
print(df.head())

# 3. 数据描述
# 3.1 缺失值处理
# 3.1.1 删去缺失值超过50%的数列
null_count = df.isnull().sum()
null_percentage = round((df.isnull().sum()/df.shape[0])*100, 2)
null_df = pd.DataFrame({'column_name' : df.columns,'null_count' : null_count,'null_percentage': null_percentage})
null_df.reset_index(drop = True, inplace = True)
null_df.sort_values(by = 'null_percentage', ascending = False)
columns_to_be_deleted = null_df[null_df['null_percentage'] > 50].column_name.to_list()
df.drop(columns = columns_to_be_deleted, inplace = True)

# 3.1.2 删去无意义的行列
df.drop(columns=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',
                 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11',
                 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',
                 'EXT_SOURCE_2', 'EXT_SOURCE_3'], inplace = True)
# 3.1.3 用众数填补分类变量缺失值
#plt.subplots(1,2 ,figsize = (20,8))
#print(df['OCCUPATION_TYPE'].value_counts())
df['OCCUPATION_TYPE'].fillna(value = 'Laborers', inplace = True)
#print(df['NAME_TYPE_SUITE'].value_counts())
df['NAME_TYPE_SUITE'].fillna(value = 'Unaccompanied', inplace = True)

# 3.2 异常值检测
#plt.subplot(121)
#sns.distplot(df[df['AMT_ANNUITY'] <= 61704.0].AMT_ANNUITY)
#pltname = 'Distplot of ' + 'AMT_ANNUITY'
#plt.title(pltname)
#plt.subplot(122)
#sns.boxplot(df[df['AMT_ANNUITY'] <= 61704.0].AMT_ANNUITY)
#pltname = 'Boxplot of ' + 'AMT_ANNUITY'
#plt.title(pltname)
#plt.tight_layout(pad = 4)
#plt.show()


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
                                 'EMERGENCYSTATE_MODE'])

df['CODE_GENDER'] = le.fit_transform(df['CODE_GENDER'])
df['FLAG_OWN_CAR'] = le.fit_transform(df['FLAG_OWN_CAR'])
df['FLAG_OWN_REALTY'] = le.fit_transform(df['FLAG_OWN_REALTY'])

# 3.3 用中位数填补分级变量缺失值
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

# 5.3 决策树模型
clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=100,random_state=0)
clf_dt.fit(X_train, y_train)

# 6. 模型评估和可视化
# 6.1 模型评估
selected_feature_names = X.columns[selector.get_support()]
evaluate_model(clf_rf, X_test, y_test, model_name='RandomForest', selected_features=selected_feature_names)
evaluate_model(clf_xgb, X_test, y_test, model_name='XGBoost', selected_features=selected_feature_names)
evaluate_model(clf_dt, X_test, y_test, model_name='DecisionTree', selected_features=selected_feature_names)
plt.show()

# 6.2 决策树绘制
fig = plt.figure(figsize=(25,20))
text_representation = tree.export_text(clf_dt,
                                       feature_names=['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'NAME_CONTRACT_TYPE_Cash loans',  
                                                      'NAME_HOUSING_TYPE_House / apartment', 'FLAG_EMP_PHONE', 
                                                      'NAME_TYPE_SUITE_Unaccompanied', 'REGION_RATING_CLIENT_W_CITY', 
                                                      'REGION_RATING_CLIENT', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 
                                                      'FLAG_OWN_REALTY', 'NAME_FAMILY_STATUS_Married', 'CNT_FAM_MEMBERS', 
                                                      'CREDIT_TERM', 'EMERGENCYSTATE_MODE_No', 'NAME_INCOME_TYPE_Working', 
                                                      'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Laborers', 'AMT_ANNUITY', 
                                                      'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_MODE'])
print(text_representation)
with open(os.path.join('C:/Users/11746/Desktop', "decistion_tree.txt"), "w") as fout:
    fout.write(text_representation)
tree.plot_tree(clf_dt, 
               feature_names=['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'NAME_CONTRACT_TYPE_Cash loans',  
                              'NAME_HOUSING_TYPE_House / apartment', 'FLAG_EMP_PHONE', 
                              'NAME_TYPE_SUITE_Unaccompanied', 'REGION_RATING_CLIENT_W_CITY', 
                              'REGION_RATING_CLIENT', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 
                              'FLAG_OWN_REALTY', 'NAME_FAMILY_STATUS_Married', 'CNT_FAM_MEMBERS', 
                              'CREDIT_TERM', 'EMERGENCYSTATE_MODE_No', 'NAME_INCOME_TYPE_Working', 
                              'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Laborers', 'AMT_ANNUITY', 
                              'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_MODE'], 
               class_names=['TARGET'],
               filled=True,
               max_depth=3)



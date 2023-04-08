from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                           confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import os

result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def evaluate_model(model, X_test, y_test, model_name, selected_features, result_dir='result'):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc_score = accuracy_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {acc_score:.4f}")
    print(f"Precision score: {prec_score:.4f}")
    print(f"Recall score: {rec_score:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Confusion matrix:")
    print(conf_matrix)

    feature_importances = pd.DataFrame({'feature': selected_features,
                                        'importance': model.feature_importances_})
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_importances['feature'], feature_importances['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top 20 Important Features ({model_name})')
    plt.savefig(os.path.join(result_dir, f'top_20_features_{model_name}.png'), bbox_inches='tight')
    plt.show()

'''
def evaluate_model(model, X_test, y_test, model_name, selected_features, result_dir='result'):    # 评估模型的准确率、精确率、召回率和F1-score
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 打印评估指标
    print("Accuracy score: {:.4f}".format(accuracy))
    print("Precision score: {:.4f}".format(precision))
    print("Recall score: {:.4f}".format(recall))
    print("F1 score: {:.4f}".format(f1))
    print("Confusion matrix:")
    print(cm)

    # 绘制 ROC 曲线
    y_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f'{model_name}_roc_curve.png'))
    plt.show()
    # 可视化并保存特征重要性
    feature_importances = pd.DataFrame({'feature': selected_features,
                                        'importance': model.feature_importances_})
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_importances['feature'][:20], feature_importances['importance'][:20])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top 20 Important Features for {model_name}')
    plt.savefig(os.path.join(result_dir, f'{model_name}_top_20_features_importance.png'), bbox_inches='tight')
    plt.show()
'''
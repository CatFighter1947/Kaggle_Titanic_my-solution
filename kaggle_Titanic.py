import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Загрузка данных
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
gender_submission_df = pd.read_csv("gender_submission.csv")


# 2. Исследование данных (EDA)
# print("Первые 5 строк обучающего набора данных:")
# print(train_df.head())
# print("\nОписание обучающего набора данных:")
# print(train_df.describe())
# print("\nИнформация об обучающем наборе данных:")
# print(train_df.info())


# 3. Визуализация данных (примеры)
sns.countplot(x='Survived', data=train_df)
plt.title('Распределение выживших и погибших')
plt.show()

sns.histplot(train_df['Age'].dropna(), kde=True)
plt.title('Распределение возраста')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Выживаемость в зависимости от класса')
plt.show()

# Проверим, имеются ли пропущенные данные
msno.matrix(train_df)
msno.matrix(test_df)

#plt.show() # Имеются, особенно в Age и Cabin

# 4. Предобработка данных
# 4.1 Обработка пропущенных значений

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median()) # inplace=True вызывает предупреждение
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Заполняем Embarked самой частой категорией
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())  #Заполняем пропуски в Fare в test_df

# Удаляем столбец Cabin (слишком много пропущенных значений)
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# 4.2 Кодирование категориальных признаков
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex']) # fit + transform
test_df['Sex'] = label_encoder.transform(test_df['Sex']) # fit, т.к. Encoder уже обученный

train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])
test_df['Embarked'] = label_encoder.transform(test_df['Embarked'])

# Дополнительно:  Создание dummy-переменных (One-Hot Encoding) для Embarked и Pclass
train_df = pd.get_dummies(train_df, columns=['Embarked', 'Pclass'], drop_first=True) # drop_first для избежания мультиколлинеарности
test_df = pd.get_dummies(test_df, columns=['Embarked', 'Pclass'], drop_first=True)

# 4.3 Удаление ненужных столбцов
train_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_PassengerId = test_df['PassengerId'] #Сохраняем PassengerId для submission
test_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)



# 5. Масштабирование признаков
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X = scaler.fit_transform(X) # Обучаем на тренировочном наболе и тут же масштабируем его

X_test = scaler.transform(test_df)  #Масштабируем тестовый набор


# 6. Разделение данных на обучающий и проверочный наборы
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Разделим обущающую выборку на еще двe: псевдообучающую и псевдотестовую для "внутренней" проверки

# 7. Обучение и оценка моделей (ROC curves) на псевдо выборках
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score

models = {
    #"GaussNB": GaussianNB(),
    #"Ada": AdaBoostClassifier(random_state=42),
    #"SVC": SVC(random_state=42, kernel='linear', probability=True), #7656
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),#7703
    "Random Forest": RandomForestClassifier(random_state=42), #7512
    "Gradient Boosting": GradientBoostingClassifier(random_state=42) #7799
}

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--') #Диагональная линия для сравнения
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Вероятности класса 1
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.legend()
plt.show()


# 8. Предсказание на тестовом наборе и создание файла submission.csv
# Моделью с наибольшим AUC оказался Random Forest, но контест выдает, что самы успешный - GradientBoostingClassifier

final_model = models["Gradient Boosting"]

# Попробовал сделать подбор гиперпараметров, но прироста в контесте это не дало
# # Определение сетки параметров для GridSearch
# param_grid = {
#     'n_estimators': [100, 200, 300, 400],  # Количество деревьев
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Скорость обучения
#     'max_depth': [3, 4, 5, 6],  # Максимальная глубина дерева
#     'min_samples_split': [2, 4, 6, 8],  # Минимальное количество выборок для разделения узла
#     'min_samples_leaf': [1, 2, 3, 4], # Минимальное количество выборок в листе
#     'subsample': [0.7, 0.8, 0.9, 1.0] # Доля выборок для обучения каждого дерева
# }
#
# grid_search = RandomizedSearchCV(final_model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)  # verbose=2 для отображения прогресса
# #cv =3 -> 3-fold cross validation
# #n_jobs=-1 -> использовать все ядра процессора
#
# grid_search.fit(X, y)

# #Вывод лучших параметров и результатов
# print("Лучшие параметры:", grid_search.best_params_)
# print("Лучшая оценка (accuracy):", grid_search.best_score_)


# grid_search.best_estimator_.fit(X, y)
# test_predictions = grid_search.best_estimator_.predict(X_test)


final_model.fit(X, y)
test_predictions = final_model.predict(X_test)

submission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)
print("\nФайл submission.csv успешно создан.")

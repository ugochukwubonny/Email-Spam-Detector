import pandas as pd # type: ignore
import numpy as np
import spacy

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import  CountVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.metrics import classification_report, ConfusionMatrixDisplay # type: ignore

df = pd.read_csv(r'accessbank_email.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df.text, df.spam, test_size=0.2, random_state=42
)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


vector = CountVectorizer()
X_train_cv = vector.fit_transform(X_train)

vector.vocabulary_


x_train_np = X_train_cv.toarray()

np.where(x_train_np[0]!=0)

x_train_np[0][290]

model = MultinomialNB()
model.fit(X_train_cv, y_train)

x_test_cv = vector.transform(X_test)

y_pred = model.predict(x_test_cv)
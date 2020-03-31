import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train_data = np.genfromtxt('train_samples.txt', delimiter = '\t', dtype = None, encoding = 'utf-8', names = ('id', 'text'),
                           comments = None)

train_labels = np.genfromtxt('train_labels.txt', delimiter='\t', dtype = None, names = ('id', 'label'))

test_data = np.genfromtxt('test_samples.txt', delimiter = '\t', dtype = None, encoding = 'utf-8', names = ('id', 'text'),
                          comments = None)

validation_data = np.genfromtxt('validation_samples.txt', delimiter='\t', dtype = None, encoding='utf-8',
                                names = ('id', 'text'), comments = None)
validation_labels = np.genfromtxt('validation_labels.txt', delimiter = '\t', dtype = None, names = ('id', 'label'))

for x in range(len(train_data)):
    train_data[x][0] = train_labels[x][1]

for x in range(len(validation_data)):
    validation_data[x][0] = validation_labels[x][1]

train_data_text = np.append(train_data['text'], validation_data['text'])
train_data_labels = np.append(train_data['id'], validation_data['id'])

# show shape of training data
#cv = CountVectorizer()
#word_count = cv.fit_transform(train_data_text)
#print(word_count.shape)

X_train = train_data_text
y_train = train_data_labels

X_test = test_data['text']

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#MNB CLASSIFICATION
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(y_pred)

f = open("predictions.txt","w+")
f.write('id')
f.write(',')
f.write('label')
f.write('\n')
for i in range(len(test_data)):
     f.write(np.array2string(test_data[i][0]))
     f.write(',')
     f.write(np.array2string(y_pred[i]))
     f.write('\n')
f.close()

from sklearn.metrics import accuracy_score
# Evaluate accuracy
#print(accuracy_score(y_test, y_pred))
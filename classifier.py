import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from smart_open import open
import pickle

class ClassifierFindU:
    def __init__(self, values, label, algor, user_id):
        self.S3_PATH = os.environ['s3_path_model']
        self.ann_classifier(values, label, algor, user_id)

    # def knn_classifier(self, features, labels, k_nn, algor, user_id):
    #     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    #     #for e in range(1,iteration_nn,2):
    #     knn = KNeighborsClassifier(n_neighbors=k_nn)

    #     knn.fit(X_train, y_train)

    #     y_pred = knn.predict(X_test)
    #     str_return = f"Accuracy: {metrics.accuracy_score(y_test, y_pred)} for {k_nn}NN"

    #     #knn = KNeighborsClassifier(n_neighbors=k_nn)
    #     #knn.fit(X_train)

    #     knn_model_p = open(f'{self.S3_PATH}knn_model_{algor}_{user_id}', 'wb')
    #     pickle.dump(knn, knn_model_p)
    #     return str_return
    #     #return knn.predict([[-23,-47.102131,1597667800]])

    def ann_classifier(self, feature, labels, algor, user_id):
        X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        DROPOUT = Dropout(0.3)
        #LEAKY_RELU = LeakyReLU(0.2)
        model = Sequential()
        model.add(Dense(4,activation='relu'))
        #model.add(LEAKY_RELU)
        model.add(DROPOUT)
        model.add(Dense(3,activation='relu'))
        #model.add(LEAKY_RELU)
        model.add(DROPOUT)

        model.add(Dense(5,activation='softmax'))
        model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        model.fit(x=X_train,y=y_train,epochs=300)

        predictions = model.predict_classes(X_test)

        model.summary()
        print(classification_report(y_test,predictions))
        print(confusion_matrix(y_test,predictions))

        print(model.predict_classes([[-23.143487,-47.2597698, 15.86346]]))
        # id = 1231241
        # with smart_open.open(f'{self.S3_PATH}ann_model_{algor}_{user_id}', 'wb') as output:
        #     output.write(model)


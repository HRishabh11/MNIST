#running the codes

from cnn_funcs import load_data, model_gen, evaluate_model,test_accuracy
X_train,y_train,X_test,y_test = load_data()
classifier = model_gen()
classifier_fit, X_train_pred = evaluate_model(classifier,X_train,y_train)
classifier.summary()
test_accuracy(classifier, X_test, y_test)

classifier.save('model.h5')
print('Model Saved')
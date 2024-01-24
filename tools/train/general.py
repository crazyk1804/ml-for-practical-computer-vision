import tensorflow as tf


def evaluate_classifier(model, dataset):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    for images, labels in dataset:
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    
    return test_loss.result(), test_accuracy.result()


def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset)
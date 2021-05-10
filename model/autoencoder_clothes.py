import numpy as np
from cv2 import cv2
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score

autoencoder = load_model('./weights/autoencoder_fashion_data.h5')

# Get encoder layer from trained model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
X_train_clothes = np.load('./weights/X_train_clothes.npy')
y_train_clothes = np.load('./weights/y_train_clothes.npy')


def retrieve_closest_images(test_element, test_label, num_images, n_samples=10):
    learned_codes = encoder.predict(X_train_clothes)
    learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                          learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])

    test_code = encoder.predict(np.array([test_element]))
    test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])

    distances = []

    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train_clothes).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    kept_indexes = sorted_indexes[:n_samples]

    score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]), np.array([sorted_distances[:n_samples]]))

    print("Average precision ranking score for tested element is {}".format(score))

    original_image = test_element
    original_image = np.array(original_image, dtype='float32')
    #cv2.imshow('', original_image)
    retrieved_images = X_train_clothes[int(kept_indexes[0]), :]
    retrieved_images = np.array(retrieved_images, dtype='float32')
    #cv2.resize(array, (1, 10000), interpolation=cv.INTER_LINEAR)
    for i in range(1, n_samples):
        retrieved_images = np.hstack((retrieved_images, X_train_clothes[int(kept_indexes[i]), :]))
    #cv2.imshow('',retrieved_images)
    #cv2.waitKey(0)

    cv2.imwrite('test_results/original_image'+str(num_images)+'.jpg',  255 * cv2.resize(original_image, (0,0), fx=3, fy=3) )
    cv2.imwrite('test_results/retrieved_results'+str(num_images)+'.jpg', 255 * cv2.resize(retrieved_images, (0,0), fx=2, fy=2))
    print("done")

#retrieve_closest_images(X_train_clothes[400], y_train_clothes[400])
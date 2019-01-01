from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from collections import Counter
import coremltools
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier

def crop_single(index):
    im = Image.open("non_contrarie.bmp")

    x_start = 70
    y_start = 50
    x_delta = 100
    y_delta = 100
    row_max = 19
    col_max = 10

    result = []

    for i in range(19):

        if i == index:
            current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
            current_region = im.crop(current_box)
            current_region = current_region.convert("L")
            current_region.show()

            pix = np.array(current_region)
            #pix = pix.reshape((test.shape[1] * test.shape[2]))
            pix = pix.reshape((current_region.size[0],) + current_region.size[1:])

            #pix = pix.reshape((pix.shape[0], pix.shape[1] * pix.shape[2]))
            #print("Pix shape {}".format(pix.shape))
            result.append(pix)

            break

        y_start = y_start + 120

    result = np.array(result)
    result = result.reshape(result.shape[1] * result.shape[1],)
    #result = result.reshape((result.shape[1] * result.shape[2]))
    print("Test shape {}".format(result.shape))
    return result


im = Image.open("non_contrarie.bmp")

x_start = 70
y_start = 50
x_delta = 100
y_delta = 100
row_max = 20
col_max = 11

labels_array = []
models_array = []
raw_labels_array = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]

for i in range(row_max):
    for j in range(col_max):
        current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
        current_region = im.crop(current_box)
        current_region = current_region.convert('L')
        #current_region = current_region.convert()

        pix = np.array(current_region)
        pix = pix.reshape((current_region.size[0] * current_region.size[1],))
        #pix = pix.reshape((pix.shape[0], pix.shape[1] * pix.shape[2]))

        models_array.append(pix)
        labels_array.append(raw_labels_array[i])

        x_start = x_start + 120

    x_start = 70
    y_start = y_start + 120

models_array = np.array(models_array)
#models_array = models_array.reshape((models_array.shape[0], models_array.shape[1] * models_array.shape[2]))
print("models Array shape {}".format(models_array.shape))
print("Labels Array shape {}".format(np.array(labels_array).shape))

zipped_array = np.array(list(zip(models_array, labels_array)))
#print("Zipped Array {}".format(zipped_array))

#clf = tree.DecisionTreeClassifier().fit(models_array, labels_array)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, min_samples_leaf=1)
clf.fit(models_array, labels_array)

#svmClf = svm.SVC(gamma='auto')
#svmClf = svmClf.fit(models_array, labels_array)
#print(svmClf.predict(models_array))

#print(clf.feature_importances_)
core_ml_model = coremltools.converters.sklearn.convert(clf)
core_ml_model.save("handWritten.mlmodel")
print(clf.predict([crop_single(6)]))
#core_ml_model = coremltools.converters.sklearn.convert(svmClf)
#core_ml_model.save("handWritten.mlmodel")
#crop_single(6)


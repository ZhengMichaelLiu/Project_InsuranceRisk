import csv
import numpy as np
import statistics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

start_time = time.time()
# Read the training data in
store_database = []
training_database = []
with open('training.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # print(','.join(row))
        store_database.append(row)

training_database = [line[1:] for line in store_database[1:]]
# print (training_database)


# clean the data
# First remove the data without a label
clean_1 = []
for each_data in training_database:
    if each_data[-1] != '':
        clean_1.append(each_data)

# Second change the prod_info_2 to number
prod_info_2_list = []
prod_info_2_list_check = []
index = 1
for each_data in clean_1:
    if each_data[1] != '' and each_data[1] not in prod_info_2_list_check:
        prod_info_2_list_check.append(each_data[1])
        prod_info_2_list.append([each_data[1], index])
        index = index + 1

for each_data in clean_1:
    if each_data[1] != '':
        for each_tuple in prod_info_2_list:
            if each_data[1] == each_tuple[0]:
                each_data[1] = each_tuple[1]
# print(clean_1)
used_for_testing_a1 = []
for each_t in prod_info_2_list:
    used_for_testing_a1.append(each_t[0])

# print(used_for_testing_a1)
# Third convert the string matrix to float matrix
for i in range(0, len(clean_1)):
    for j in range(0, len(clean_1[0])):
        if clean_1[i][j] != '':
            clean_1[i][j] = float(clean_1[i][j])
# print(clean_1)

# Forth fill in the missing values
    # Separate groups based on the label
Group_1 = []
Group_2 = []
Group_3 = []
Group_4 = []
Group_5 = []
Group_6 = []
Group_7 = []
Group_8 = []

for each_data in clean_1:
    if each_data[-1] == 1:
        Group_1.append(each_data)
    if each_data[-1] == 2:
        Group_2.append(each_data)
    if each_data[-1] == 3:
        Group_3.append(each_data)
    if each_data[-1] == 4:
        Group_4.append(each_data)
    if each_data[-1] == 5:
        Group_5.append(each_data)
    if each_data[-1] == 6:
        Group_6.append(each_data)
    if each_data[-1] == 7:
        Group_7.append(each_data)
    if each_data[-1] == 8:
        Group_8.append(each_data)

    # Get average or mode of each attribute in each group
Groups = []
Groups.append(Group_1)
Groups.append(Group_2)
Groups.append(Group_3)
Groups.append(Group_4)
Groups.append(Group_5)
Groups.append(Group_6)
Groups.append(Group_7)
Groups.append(Group_8)

num_attris = len(clean_1[0]) - 1    # 126

total_existing_attris = []

for i in range(0, 8):
    each_group_existing_attris = []
    for j in range(0, num_attris):
        small_group = []
        for each_member in Groups[i]:
            if each_member[j] != '':
                small_group.append(each_member[j])
        each_group_existing_attris.append(small_group)
    total_existing_attris.append(each_group_existing_attris)
# print(total_existing_attris)

    # find what value should be set to empty attributes in each group
total_replace_attris = []
for i in range(0, 8):
    each_group_replace_attris = []
    for j in range(0, num_attris):
        if j in [3, 7, 8, 9, 10, 11, 14, 16, 28, 33, 34, 35, 36, 37, 46, 51, 60, 68]:
            small_valid_group = []
            firstq = np.percentile(np.array(total_existing_attris[i][j]), 25)
            thirdq = np.percentile(np.array(total_existing_attris[i][j]), 75)
            for each_element in total_existing_attris[i][j]:
                if each_element <= thirdq and each_element >= firstq:
                    small_valid_group.append(each_element)
            result = statistics.median(small_valid_group)
            each_group_replace_attris.append(result)
        else:
            result = statistics.mode(total_existing_attris[i][j])
            each_group_replace_attris.append(result)
    total_replace_attris.append(each_group_replace_attris)

# print(len(total_replace_attris[7]))

    # replace empty values
for i in range(0, len(clean_1)):
    if clean_1[i] in Group_1:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[0][j]
    if clean_1[i] in Group_2:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[1][j]
    if clean_1[i] in Group_3:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[2][j]
    if clean_1[i] in Group_4:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[3][j]
    if clean_1[i] in Group_5:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[4][j]
    if clean_1[i] in Group_6:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[5][j]
    if clean_1[i] in Group_7:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[6][j]
    if clean_1[i] in Group_8:
        for j in range(0, num_attris):
            if clean_1[i][j] == '':
                clean_1[i][j] = total_replace_attris[7][j]

# Finish cleaning training data

# print(clean_1)

# Get training features
training_features = []

for i in range(0, len(clean_1)):
    current_row = []
    for j in range(0, num_attris):
        current_row.append(clean_1[i][j])
    training_features.append(current_row)

# Get training labels
training_labels = []
for i in range(0, len(clean_1)):
    training_labels.append(clean_1[i][-1])

# print(training_features)
# print(training_labels)

# print("--- %s seconds ---" % (time.time() - start_time))

# Read the testing data in
store_database = []
test_database = []
with open('testing.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # print(','.join(row))
        store_database.append(row)

ID_list = [line[0] for line in store_database[1:]]

test_database = [line[1:] for line in store_database[1:]]

clean_2 = []
for each_data in test_database:
    clean_2.append(each_data)

for each_data in clean_2:

    if each_data[1] != '' and each_data[1] in used_for_testing_a1:
        for each_tuple in prod_info_2_list:
            if each_data[1] == each_tuple[0]:
                each_data[1] = each_tuple[1]
    elif each_data[1] != '' and each_data[1] not in used_for_testing_a1:
        each_data[1] = 0

for i in range(0, len(clean_2)):
    for j in range(0, len(clean_2[0])):
        if clean_2[i][j] != '':
            clean_2[i][j] = float(clean_2[i][j])

test_existing_attris = []
for i in range(0, num_attris):
    col_testing = []
    for each_data in clean_2:
        if each_data[i] != '':
            col_testing.append(each_data[i])
    test_existing_attris.append(col_testing)


test_replace_attris = []

for i in range(0, num_attris):
    if i in [3, 7, 8, 9, 10, 11, 14, 16, 28, 33, 34, 35, 36, 37, 46, 51, 60, 68]:
        col_valid_group = []
        firstq = np.percentile(np.array(test_existing_attris[i]), 25)
        thirdq = np.percentile(np.array(test_existing_attris[i]), 75)
        for each_element in test_existing_attris[i]:
            if each_element <= thirdq and each_element >= firstq:
                col_valid_group.append(each_element)
        result = statistics.median(col_valid_group)
        test_replace_attris.append(result)
    else:
        result = statistics.mode(test_existing_attris[i])
        test_replace_attris.append(result)

for i in range(0, len(clean_2)):
    for j in range(0, num_attris):
        if clean_2[i][j] == '':
                clean_2[i][j] = test_replace_attris[j]

testing_features = []

for i in range(0, len(clean_2)):
    current_row = []
    for j in range(0, num_attris):
        current_row.append(clean_2[i][j])
    testing_features.append(current_row)
# print(testing_features)


# Create validation sets
num_train_data = len(training_features)

train_feature_1 = []
train_label_1 = []
validation_feature_1 = []
validation_label_1 = []

train_feature_2 = []
train_label_2 = []
validation_feature_2 = []
validation_label_2 = []

train_feature_3 = []
train_label_3 = []
validation_feature_3 = []
validation_label_3 = []

train_feature_4 = []
train_label_4 = []
validation_feature_4 = []
validation_label_4 = []

train_feature_5 = []
train_label_5 = []
validation_feature_5 = []
validation_label_5 = []

# validation 1
for i in range(0, int((num_train_data / 5) * 4)):
    train_feature_1.append(training_features[i])
    train_label_1.append(training_labels[i])
for i in range(int((num_train_data / 5) * 4), num_train_data):
    validation_feature_1.append(training_features[i])
    validation_label_1.append(training_labels[i])

# validation 2
for i in range(0, int((num_train_data / 5) * 3)):
    train_feature_2.append(training_features[i])
    train_label_2.append(training_labels[i])
for i in range(int((num_train_data / 5) * 4), num_train_data):
    train_feature_2.append(training_features[i])
    train_label_2.append(training_labels[i])

for i in range(int((num_train_data / 5) * 3), int((num_train_data / 5) * 4)):
    validation_feature_2.append(training_features[i])
    validation_label_2.append(training_labels[i])

# validation 3
for i in range(0, int((num_train_data / 5) * 2)):
    train_feature_3.append(training_features[i])
    train_label_3.append(training_labels[i])
for i in range(int((num_train_data / 5) * 3), num_train_data):
    train_feature_3.append(training_features[i])
    train_label_3.append(training_labels[i])

for i in range(int((num_train_data / 5) * 2), int((num_train_data / 5) * 3)):
    validation_feature_3.append(training_features[i])
    validation_label_3.append(training_labels[i])

# validation 4
for i in range(0, int((num_train_data / 5) * 1)):
    train_feature_4.append(training_features[i])
    train_label_4.append(training_labels[i])
for i in range(int((num_train_data / 5) * 2), num_train_data):
    train_feature_4.append(training_features[i])
    train_label_4.append(training_labels[i])

for i in range(int((num_train_data / 5) * 1), int((num_train_data / 5) * 2)):
    validation_feature_4.append(training_features[i])
    validation_label_4.append(training_labels[i])

# validation 5

for i in range(int((num_train_data / 5) * 1), num_train_data):
    train_feature_5.append(training_features[i])
    train_label_5.append(training_labels[i])

for i in range(0, int((num_train_data / 5) * 1)):
    validation_feature_5.append(training_features[i])
    validation_label_5.append(training_labels[i])

# Create Parameter List
layer_sizes = [(150, 150, 150), (200, 200, 200), (50, 50, 50, 50), (100, 100, 100, 100)]
activ_function = ['logistic', 'tanh', 'relu']
alpha = [0.0005, 0.0001, 0.00005]
max_iter = [250, 300, 350]
learning_rate = [0.0005, 0.001]

Parameter_list = []
for each_layer_size in layer_sizes:
    for each_activ_function in activ_function:
        for each_alpha in alpha:
            for each_max_iter in max_iter:
                for each_learning_rate in learning_rate:
                    Parameter_list.append([each_layer_size, each_activ_function, each_alpha, each_max_iter, each_learning_rate])

acc_list = []

for i in range(0, len(Parameter_list)):
    current_parameter  = Parameter_list[i]
    clf = MLPClassifier(hidden_layer_sizes=current_parameter[0], max_iter=current_parameter[3], alpha=current_parameter[2],learning_rate_init=current_parameter[4],activation=current_parameter[1])
    clf.fit(train_feature_1, train_label_1)
    p1 = clf.predict(validation_feature_1)
    acc1 = accuracy_score(p1, validation_label_1)

    clf.fit(train_feature_2, train_label_2)
    p2 = clf.predict(validation_feature_2)
    acc2 = accuracy_score(p2, validation_label_2)

    clf.fit(train_feature_3, train_label_3)
    p3 = clf.predict(validation_feature_3)
    acc3 = accuracy_score(p3, validation_label_3)

    clf.fit(train_feature_4, train_label_4)
    p4 = clf.predict(validation_feature_4)
    acc4 = accuracy_score(p4, validation_label_4)

    clf.fit(train_feature_5, train_label_5)
    p5 = clf.predict(validation_feature_5)
    acc5 = accuracy_score(p5, validation_label_5)

    avg_acc = (acc1 + acc2 + acc3 +  acc4 + acc5) / 5

    acc_list.append(avg_acc)
    print('Parameter: ', Parameter_list[i], ' Accuracy: ', avg_acc)

best_acc = max(acc_list)
index_of_best_acc = [i for i, j in enumerate(acc_list) if j == best_acc]

best_parameter = Parameter_list[index_of_best_acc[0]]

final_clf = MLPClassifier(hidden_layer_sizes=best_parameter[0], max_iter=best_parameter[3], 
                          alpha=best_parameter[2],learning_rate_init=best_parameter[4],
                          activation=best_parameter[1])
final_clf.fit(training_features, training_labels)
final_prediction = final_clf.predict(testing_features)

final_answer = []
for i in range(0, len(ID_list)):
    final_answer.append([str(ID_list[i]), str(int(final_prediction[i]))])

with open('solution.csv','w') as f:
    w = csv.writer(f)
    w.writerow(['Id','Response'])
    w.writerows(final_answer)
## Getting Started With TensorFlow

### TensorFlow Programming Stack

![](https://raw.githubusercontent.com/Diagon-Alley/TensorFlowDocs/master/static/tensorflow_programming_environment.png)

Note that tensorflow official documnetation strongly recommends focussing on the following 2 high-level APIs:
* Estimators
* Datasets

Steps to get started with a simple machine learning problem using the sample programs of tensorflow

* Import and parse the data
* Create feature columns to describe the data
* Select the type of model
* Train the model
* Evaluate the model's effectiveness
* Let the trained model make predictions

To start with obviously the entire dataset will be separated into training and test sets.

In the sample code of tensorflow, the `premade_estimators.py` forms the bulk of the entire program. This program relies on the `load_data` function in the adjacent `iris_data.py` file to read in and parse the training and test set.

```python
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# The same dataset has separate training and test urls for training and test data

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

...

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Keras is an open source machine learning library. tf.keras is a tensorflow implementation of keras
    # The tf.keras.utils.get_file function is a convenience function that simply copies a remote CSV file
    # to a local file system. Among the keyword args fname is the filename and the origin is the url origin

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)
```

The call to `load_data()` would return two `(feature, label)` pairs for training and test sets respectively

A pandas **`DataFrame`** is a table with named column headers and numbered rows. The feature returned by `load_data` are packed in `DataFrames`.

A **`feature column`** is a data-structure that tells your model how to interpret the data in each feature. Note that this is a key element in the entire problem. Sometimes we may require to use the same features as marked in the original raw dataset quite literally and at other times we may be need to change the feature column to make it more suitable to the learning model.

From a code perspective we build a list of `feature_column` objects using the `tf.feature_column` module. Each object describes an input to the model. 

```python
# Create feature columns for all features.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

### Specifying a model

To specify a model type we need to instantiate an **Estimator** class. Tensor flow provides 2 categories of estimators:

* pre-made Estimators: Estimators written beforehand.
* custom Estimators: Estimators to be written (atleast partially) by the developers.

To implement the neural network the `premade_estimators.py` program uses a premade estimator called `tf.estimator.DNNClassifier`. The estimator builds a neural network that classifies examples.

```python
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)

        # recall that my_feature_columns is a list of all the features using tf.feature_column with a key parameter
        # hidden_units is simply the hidden layers. 10 in the first hidden layer and 10 in the second 
        # hidden layer
        # n_classes as you may have guessed is the number of classes in which to classify
```

Another important idea is that there is an optional keyword argument called `optimizer` The `optimizer` simply controls how our model will train. There is also `learning rate`

### Training the model

Now we have specified how the network architecture should be but have not trained the model. Now we need to train the model.

```python
    classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)
```

The keyword argument steps is actually the argument that controls the number of iterations while having to train the model. The default value in this case is simply 1000. How many iterations a model requires again depends on experience.

The `input_fn` parameter identifies the function that supplies the training data. Recall that we have simply built the training NN architecture. We have not trained it yet. The input function is simply required to do that. Here is the method signature of the input function:

```python
def train_input_fn(features, labels, batch_size):
```

We pass the following parameters to the train_input_fn function:

* `train_feature`: A python dictionary in which each key is the name of the feature and each value is the array containing the values for each example in the training set

* `train_label`: Simply an array containing the values of the label for every example.

* `arg.bactch_size`: A **batch** is the set of examples used in one iteration. The number of examples in a batch is the batch size. In case of stochastic gradient descent the batch size is 1. Obviously we know that in general gradient descent at one iteration only one example is used for training.


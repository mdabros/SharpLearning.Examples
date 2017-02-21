SharpLearning.Examples
======================

SharpLearning.Examples contains code examples to show how to use the machine learning library [SharpLearning](https://github.com/mdabros/SharpLearning)

The examples are structured in a unittest project, where each test is a separate example. 
There are no assertions and the unit test framework is only used as an easy way of executing the examples.

An example showing how to read data, create a RegressionDecisionTreeLearner and learn a
RegressionDecisionTreeModel can be seen below:

```c#
[TestMethod]
public void RegressionLearner_Learn_And_Predcit()
{
    // Use StreamReader(filepath) when running from filesystem
    var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
    var targetName = "quality";

    // read feature matrix
    var observations = parser.EnumerateRows(c => c != targetName)
        .ToF64Matrix();

    // read regression targets
    var targets = parser.EnumerateRows(targetName)
        .ToF64Vector();

    // create learner
    var learner = new RegressionDecisionTreeLearner(maximumTreeDepth: 5);

    // learns a RegressionRandomForestModel
    var model = learner.Learn(observations, targets);

    // use the model to predict the training data
    var predictions = model.Predict(observations);
}
```

Currently there are only examples showing general usage of the learners and models. 
Specific examples for the individual learner/model types is planned. However, except from different hyperparameters,
the the individual learner types are used in exactly the same way. Same example as above but with a RandomForestLearner instead of a DecisionTreeLearner:


```c#
[TestMethod]
public void RegressionLearner_Learn_And_Predcit()
{
    // Use StreamReader(filepath) when running from filesystem
    var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
    var targetName = "quality";

    // read feature matrix
    var observations = parser.EnumerateRows(c => c != targetName)
        .ToF64Matrix();

    // read regression targets
    var targets = parser.EnumerateRows(targetName)
        .ToF64Vector();

    // create learner (RandomForest)
    var learner = new RegressionRandomForestLearner(trees: 100);

    // learns a RegressionDecisionTreeModel
    var model = learner.Learn(observations, targets);

    // use the model to predict the training data
    var predictions = model.Predict(observations);
}
```

Datasets
---------
SharpLearning.Examples contains 3 datasets in csv format for use with the examples:

**MNIST_SMALL**
- *mnist_small_train* - 1000 examples from the original MNIST training data set
- *mnist_small_test* - 1000 examples from the original MNIST test data set

The original and complete dataset can be found here:
http://yann.lecun.com/exdb/mnist/

**CIFAR10_SMALL**
- *cifar10_train_small* - 1000 examples from the original CIFAR10 training data set
- *cifar10_test_small* - 1000 examples from the original CIFAR10 test data set

The original and complete dataset can be found here:
https://www.cs.toronto.edu/~kriz/cifar.html

**Wine Quality Data Set**
- *winequality-white* - the complete white wine quality dataset

The original dataset can be found here:
http://archive.ics.uci.edu/ml/datasets/Wine+Quality

Installation
------------

Installation instructions for SharpLearning are availible in the main SharpLearning repository:
https://github.com/mdabros/SharpLearning

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers.Matrices;
using System.IO;
using System.Linq;
using SharpLearning.Neural;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Metrics.Classification;
using System.Diagnostics;

namespace SharpLearning.Examples.NeuralNets
{
    [TestClass]
    public class ClassificationNeuralNetExamples
    {
        [TestMethod]
        public void Classification_Convolutional_Neural_Net()
        {
            #region Read Data
            
            // Use StreamReader(filepath) when running from filesystem
            var trainingParser = new CsvParser(() => new StringReader(Resources.mnist_small_train));
            var testParser = new CsvParser(() => new StringReader(Resources.mnist_small_test));

            var targetName = "Class";

            var featureNames = trainingParser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex.Keys.ToArray();

            // read feature matrix (training)
            var trainingObservations = trainingParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (training)
            var trainingTargets = trainingParser.EnumerateRows(targetName)
                .ToF64Vector();

            // read feature matrix (test) 
            var testObservations = testParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (test)
            var testTargets = testParser.EnumerateRows(targetName)
                .ToF64Vector();
            #endregion

            // transform pixel values to be between 0 and 1.
            trainingObservations.Map(p => p / 255);
            testObservations.Map(p => p / 255);

            // the output layer must know the number of classes.
            var numberOfClasses = trainingTargets.Distinct().Count();

            // define the neural net.
            var net = new NeuralNet();
            net.Add(new InputLayer(width: 28, height:  28, depth: 1)); // MNIST data is 28x28x1.
            net.Add(new Conv2DLayer(filterWidth: 5, filterHeight: 5, filterCount: 32));
            net.Add(new MaxPool2DLayer(poolWidth: 2, poolHeight: 2));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(256, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SoftMaxLayer(numberOfClasses));

            // using only 10 iteration to make the example run faster.
            // using classification accuracy as error metric. This is only used for reporting progress.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 10, loss: new AccuracyLoss());
            var model = learner.Learn(trainingObservations, trainingTargets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }

        [TestMethod]
        public void Classification_Standard_Neural_Net()
        {
            #region Read Data
            // Use StreamReader(filepath) when running from filesystem
            var trainingParser = new CsvParser(() => new StringReader(Resources.mnist_small_train));
            var testParser = new CsvParser(() => new StringReader(Resources.mnist_small_test));

            var targetName = "Class";

            var featureNames = trainingParser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex.Keys.ToArray();

            // read feature matrix (training)
            var trainingObservations = trainingParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (training)
            var trainingTargets = trainingParser.EnumerateRows(targetName)
                .ToF64Vector();

            // read feature matrix (test) 
            var testObservations = testParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (test)
            var testTargets = testParser.EnumerateRows(targetName)
                .ToF64Vector();
            #endregion

            // transform pixel values to be between 0 and 1.
            trainingObservations.Map(p => p / 255);
            testObservations.Map(p => p / 255);

            // the output layer must know the number of classes.
            var numberOfClasses = trainingTargets.Distinct().Count();

            var net = new NeuralNet();
            net.Add(new InputLayer(width: 28, height: 28, depth: 1)); // MNIST data is 28x28x1.
            net.Add(new DropoutLayer(0.2));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SoftMaxLayer(numberOfClasses));

            // using only 10 iteration to make the example run faster.
            // using classification accuracy as error metric. This is only used for reporting progress.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 10, loss: new AccuracyLoss());
            var model = learner.Learn(trainingObservations, trainingTargets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }

        [TestMethod]
        public void Classification_LogisticRegression_Neural_Net()
        {
            #region Read Data
            // Use StreamReader(filepath) when running from filesystem
            var trainingParser = new CsvParser(() => new StringReader(Resources.mnist_small_train));
            var testParser = new CsvParser(() => new StringReader(Resources.mnist_small_test));

            var targetName = "Class";

            var featureNames = trainingParser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex.Keys.ToArray();

            // read feature matrix (training)
            var trainingObservations = trainingParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (training)
            var trainingTargets = trainingParser.EnumerateRows(targetName)
                .ToF64Vector();

            // read feature matrix (test) 
            var testObservations = testParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (test)
            var testTargets = testParser.EnumerateRows(targetName)
                .ToF64Vector();
            #endregion

            // transform pixel values to be between 0 and 1.
            trainingObservations.Map(p => p / 255);
            testObservations.Map(p => p / 255);

            // the output layer must know the number of classes.
            var numberOfClasses = trainingTargets.Distinct().Count();

            var net = new NeuralNet();
            net.Add(new InputLayer(width: 28, height: 28, depth: 1)); // MNIST data is 28x28x1.
            net.Add(new SoftMaxLayer(numberOfClasses)); // No hidden layers and SoftMax output layer corresponds to logistic regression classifer.

            // using only 10 iteration to make the example run faster.
            // using classification accuracy as error metric. This is only used for reporting progress.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 10, loss: new AccuracyLoss());
            var model = learner.Learn(trainingObservations, trainingTargets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }

        [TestMethod]
        public void Classification_LinearSvm_Neural_Net()
        {
            #region Read Data
            // Use StreamReader(filepath) when running from filesystem
            var trainingParser = new CsvParser(() => new StringReader(Resources.mnist_small_train));
            var testParser = new CsvParser(() => new StringReader(Resources.mnist_small_test));

            var targetName = "Class";

            var featureNames = trainingParser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex.Keys.ToArray();

            // read feature matrix (training)
            var trainingObservations = trainingParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (training)
            var trainingTargets = trainingParser.EnumerateRows(targetName)
                .ToF64Vector();

            // read feature matrix (test) 
            var testObservations = testParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (test)
            var testTargets = testParser.EnumerateRows(targetName)
                .ToF64Vector();
            #endregion

            // transform pixel values to be between 0 and 1.
            trainingObservations.Map(p => p / 255);
            testObservations.Map(p => p / 255);

            // the output layer must know the number of classes.
            var numberOfClasses = trainingTargets.Distinct().Count();

            var net = new NeuralNet();
            net.Add(new InputLayer(width: 28, height: 28, depth: 1)); // MNIST data is 28x28x1.
            net.Add(new SvmLayer(numberOfClasses)); // No hidden layers and Svm output layer corresponds to linear Svm.

            // using only 10 iteration to make the example run faster.
            // using classification accuracy as error metric. This is only used for reporting progress.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 10, loss: new AccuracyLoss());
            var model = learner.Learn(trainingObservations, trainingTargets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }
    }
}

using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.Neural;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Optimizers;

namespace SharpLearning.Examples.NeuralNets
{
    [TestClass]

    public class OptimizerExamples
    {
        [TestMethod]
        public void Classification_Convolutional_Neural_Net_Select_Optimizer_Method()
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
            net.Add(new InputLayer(width: 28, height: 28, depth: 1)); // MNIST data is 28x28x1.
            net.Add(new Conv2DLayer(filterWidth: 5, filterHeight: 5, filterCount: 32));
            net.Add(new MaxPool2DLayer(poolWidth: 2, poolHeight: 2));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(256, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SoftMaxLayer(numberOfClasses));

            // Different optimizer methods can be selected.
            // SharpLearning default is RMSProp.
            // Recommendations would be: RMSProp, Adam, Nadam or Adadelta.
            var optimizerMethod = OptimizerMethod.Nadam;

            // using only 10 iteration to make the example run faster.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 10, loss: new AccuracyLoss(), 
                optimizerMethod: optimizerMethod);

            var model = learner.Learn(trainingObservations, trainingTargets);
            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }
    }
}
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.Neural;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Examples.NeuralNets
{
    [TestClass]

    public class BatchNormalizationExamples
    {
        [TestMethod]
        public void Classification_Neural_Net_Using_BatchNormalization()
        {
            #region Read Data
            // Use StreamReader(filepath) when running from filesystem
            var trainingParser = new CsvParser(() => new StringReader(Resources.cifar10_train_small));
            var testParser = new CsvParser(() => new StringReader(Resources.cifar10_test_small));

            var targetName = "label";
            var id = "id";

            var featureNames = trainingParser.EnumerateRows(v => v != targetName && v != id).First().ColumnNameToIndex.Keys.ToArray();

            var index = 0.0;
            var targetNameToTargetValue = trainingParser.EnumerateRows(targetName)
                .ToStringVector().Distinct().ToDictionary(v => v, v => index++);

            // read feature matrix (training)
            var trainingObservations = trainingParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (training)
            var trainingTargets = trainingParser.EnumerateRows(targetName)
                .ToStringVector().Select(v => targetNameToTargetValue[v]).ToArray();

            // read feature matrix (test) 
            var testObservations = testParser
                .EnumerateRows(featureNames)
                .ToF64Matrix();
            // read classification targets (test)
            var testTargets = testParser.EnumerateRows(targetName)
                .ToStringVector().Select(v => targetNameToTargetValue[v]).ToArray();
            #endregion

            // transform pixel values to be between 0 and 1.
            trainingObservations.Map(p => p / 255);
            testObservations.Map(p => p / 255);

            // the output layer must know the number of classes.
            var numberOfClasses = trainingTargets.Distinct().Count();

            // batch normalization can be added to all layers with weights + biases.
            // Batch normalization will increase the error reduction pr. iteration
            // but will also make each iteration more slow due to the extra work.
            // Batch normalization usually has the best effect on deeper networks.
            var useBatchNorm = true;
            var net = new NeuralNet();
            net.Add(new InputLayer(width: 32, height: 32, depth: 3)); // CIFAR data is 32x32x3.
            net.Add(new Conv2DLayer(3, 3, 32) { BatchNormalization = useBatchNorm }); // activate batch normalization.
            net.Add(new MaxPool2DLayer(2, 2));
            net.Add(new DropoutLayer(0.25));
            net.Add(new Conv2DLayer(3, 3, 64) { BatchNormalization = useBatchNorm }); // activate batch normalization.
            net.Add(new Conv2DLayer(3, 3, 64) { BatchNormalization = useBatchNorm }); // activate batch normalization.
            net.Add(new MaxPool2DLayer(2, 2));
            net.Add(new DropoutLayer(0.25));
            net.Add(new DenseLayer(512) { BatchNormalization = useBatchNorm }); // activate batch normalization.
            net.Add(new DropoutLayer(0.5));
            net.Add(new SoftMaxLayer(numberOfClasses));

            // using classification accuracy as error metric. 
            // When using a validation set, the error metric 
            // is used for selecting the best iteration based on models error on the validation set.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 5, loss: new AccuracyLoss());

            var model = learner.Learn(trainingObservations, trainingTargets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }
    }
}

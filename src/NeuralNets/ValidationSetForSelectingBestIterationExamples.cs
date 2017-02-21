using SharpLearning.Examples.Properties;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using System.Linq;
using SharpLearning.Containers.Matrices;
using SharpLearning.Neural;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Metrics.Classification;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using System.Diagnostics;

namespace SharpLearning.Examples.NeuralNets
{
    [TestClass]

    public class ValidationSetForSelectingBestIterationExamples
    {
        [TestMethod]
        public void Classification_Neural_Net_Using_ValidtionSet_For_Selecting_The_best_Model()
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

            // create training validation split
            var splitter = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);
            var split = splitter.SplitSet(trainingObservations, trainingTargets);
            
            // the output layer must know the number of classes.
            var numberOfClasses = trainingTargets.Distinct().Count();

            var net = new NeuralNet();
            net.Add(new InputLayer(width: 28, height: 28, depth: 1)); // MNIST data is 28x28x1.
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new SoftMaxLayer(numberOfClasses));

            // using classification accuracy as error metric. 
            // When using a validation set, the error metric 
            // is used for selecting the best iteration based on models error on the validation set.
            var learner = new ClassificationNeuralNetLearner(net, iterations: 10, loss: new AccuracyLoss());

            var model = learner.Learn(split.TrainingSet.Observations, split.TrainingSet.Targets,//);
                split.TestSet.Observations, split.TestSet.Targets); // the validation set for estimating how well the network generalises to new data.

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(testObservations);

            Trace.WriteLine("Test Error: " + metric.Error(testTargets, predictions));
        }
    }
}

using System.Diagnostics;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Examples.Properties;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Examples.NeuralNets
{
    [TestClass]
    public class NeuralNetExamples
    {
        [TestMethod]
        public void Regression_Standard_Neural_Net()
        {
            #region Read Data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            #endregion

            // transform pixel values to be between 0 and 1.
            var minMaxTransformer = new MinMaxTransformer(0.0, 1.0);
            minMaxTransformer.Transform(observations, observations);

            var numberOfFeatures = observations.ColumnCount;

            // define the neural net.
            var net = new NeuralNet();
            net.Add(new InputLayer(inputUnits: numberOfFeatures));
            net.Add(new DropoutLayer(0.2));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SquaredErrorRegressionLayer()); 

            // using only 10 iteration to make the example run faster.
            // using square error as error metric. This is only used for reporting progress.
            var learner = new RegressionNeuralNetLearner(net, iterations: 10, loss: new SquareLoss());
            var model = learner.Learn(observations, targets);

            var metric = new MeanSquaredErrorRegressionMetric();
            var predictions = model.Predict(observations);

            Trace.WriteLine("Training Error: " + metric.Error(targets, predictions));
        }

        [TestMethod]
        public void Regression_Standard_Neural_Net_FeatureTransform_Normalization()
        {
            #region Read Data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            #endregion

            // transform pixel values to be between 0 and 1
            // and shift each feature to have a mean value of zero.
            var minMaxTransformer = new MinMaxTransformer(0.0, 1.0);
            var meanZeroTransformer = new MeanZeroFeatureTransformer();

            minMaxTransformer.Transform(observations, observations);
            meanZeroTransformer.Transform(observations, observations);

            var numberOfFeatures = observations.ColumnCount;

            // define the neural net.
            var net = new NeuralNet();
            net.Add(new InputLayer(inputUnits: numberOfFeatures));
            net.Add(new DropoutLayer(0.2));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SquaredErrorRegressionLayer());

            // using only 10 iteration to make the example run faster.
            // using square error as error metric. This is only used for reporting progress.
            var learner = new RegressionNeuralNetLearner(net, iterations: 10, loss: new SquareLoss());
            var model = learner.Learn(observations, targets);

            var metric = new MeanSquaredErrorRegressionMetric();
            var predictions = model.Predict(observations);

            Trace.WriteLine("Training Error: " + metric.Error(targets, predictions));
        }
    }
}

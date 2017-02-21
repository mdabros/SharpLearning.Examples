using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Examples.Properties;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.Neural;
using SharpLearning.Neural.Layers;
using System.Linq;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Examples.FeatureTransformations
{
    [TestClass]
    public class FeatureNormalizationExample
    {
        [TestMethod]
        public void FeatureNormalization_Normalize()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix (all columns different from the targetName)
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // create minmax normalizer (normalizes each feature from 0.0 to 1.0)
            var minMaxTransformer = new MinMaxTransformer(0.0, 1.0);

            // transforms features using the feature normalization transform 
            minMaxTransformer.Transform(observations, observations);

            // read targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            // neural net requires features to be normalize. 
            // This makes convergens much faster.
            var net = new NeuralNet();
            net.Add(new InputLayer(observations.ColumnCount)); 
            net.Add(new SoftMaxLayer(targets.Distinct().Count())); // no hidden layer and softmax output correpsonds to logistic regression
            var learner = new ClassificationNeuralNetLearner(net, new LogLoss());

            // learns a logistic regression classifier
            var model = learner.Learn(observations, targets);
        }
    }
}

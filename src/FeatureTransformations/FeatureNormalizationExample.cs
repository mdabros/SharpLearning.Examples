using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Examples.Properties;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;
using SharpLearning.InputOutput.Serialization;
using SharpLearning.Neural;
using SharpLearning.Neural.Layers;
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

            // Create neural net.
            var net = new NeuralNet();
            net.Add(new InputLayer(observations.ColumnCount)); 
            net.Add(new SquaredErrorRegressionLayer());

            // Create regression learner.
            var learner = new RegressionNeuralNetLearner(net, new SquareLoss());

            // learns a neural net regression model.
            var model = learner.Learn(observations, targets);

            // serializer for saving the MinMaxTransformer
            var serializer = new GenericXmlDataContractSerializer();

            // Serialize transform for use with model.
            // Replace this with StreamWriter for use with file system.
            var data = new StringBuilder();
            var writer = new StringWriter(data);
            serializer.Serialize(minMaxTransformer, () => writer);

            // Deserialize transform for use with model.
            // Replace this with StreamReader for use with file system.
            var reader = new StringReader(data.ToString());
            var deserializedMinMaxTransform = serializer.Deserialize<MinMaxTransformer>(() => reader);

            // Normalize observation and predict using the model.
            var normalizedObservation = deserializedMinMaxTransform.Transform(observations.Row(0));
            var prediction = model.Predict(normalizedObservation);

            Trace.WriteLine($"Prediction: {prediction}");
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;
using CntkCatalyst;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.CrossValidation.Samplers;
using SharpLearning.Examples.Integration;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Optimizers;

namespace SharpLearning.Examples.IntegrationWithOtherMLPackages
{
    [TestClass]
    public class CntkIntegrations
    {
        /// <summary>
        /// Pure SharpLearning Example for comparison.
        /// </summary>
        [TestMethod]
        public void SharpLearning_Example()
        {
            // Load data
            var (observations, targets) = DataSetUtilities.LoadWinequalityWhite();

            // transform data for neural net
            var transform = new MinMaxTransformer(0.0, 1.0);
            transform.Transform(observations, observations);

            var featureCount = observations.ColumnCount;

            // define the neural net.
            var net = new NeuralNet();
            net.Add(new InputLayer(inputUnits: featureCount));
            net.Add(new DenseLayer(32, Activation.Relu));
            net.Add(new DenseLayer(32, Activation.Relu));
            net.Add(new SquaredErrorRegressionLayer());

            // using only 10 iteration to make the example run faster.
            // using square error as error metric. This is only used for reporting progress.
            var learner = new RegressionNeuralNetLearner(net, iterations: 10, loss: new SquareLoss(),
                optimizerMethod: OptimizerMethod.Adam);

            var cv = new RandomCrossValidation<double>(10, seed: 232);
            var predictions = cv.CrossValidate(learner, observations, targets);

            Trace.WriteLine(FormatErrorString(targets, predictions));
        }

        /// <summary>
        /// Note that this uses the CntkCatalyst package, which adds layers, and model API to CNTK C#.
        /// Ongoing effort to refactor/add SharpLearning utilities to make integration with CNTK, and other ML frameworks, easier.
        /// This also serve to improve the CntkCatalyst extensions.
        /// </summary>
        [TestMethod]
        public void SharpLearning_With_Cntk_Example()
        {
            // Load data
            var (observations, targets) = DataSetUtilities.LoadWinequalityWhite();

            // transform data for neural net
            var transform = new MinMaxTransformer(0.0, 1.0);
            transform.Transform(observations, observations);

            var featureCount = observations.ColumnCount;
            var observationCount = observations.RowCount;
            var targetCount = 1;

            var inputShape = new int[] { featureCount, 1 };
            var outputShape = new int[] { targetCount };

            // Convert data to float, and wrap as minibatch data.
            var observationsFloat = observations.Data().Select(v => (float)v).ToArray();
            var observationsData = new MemoryMinibatchData(observationsFloat, inputShape, observationCount);
            var targetsFloat = targets.Select(v => (float)v).ToArray();
            var targetsData = new MemoryMinibatchData(targetsFloat, outputShape, observationCount);

            var dataType = DataType.Float;
            var device = DeviceDescriptor.CPUDevice;

            // setup input and target variables.
            var inputVariable = Layers.Input(inputShape, dataType);
            var targetVariable = Variable.InputVariable(outputShape, dataType);

            // setup name to variable
            var nameToVariable = new Dictionary<string, Variable>
            {
                { "observations", inputVariable },
                { "targets", targetVariable },
            };

            // Get cross validation folds.
            var sampler = new RandomIndexSampler<double>(seed: 24);
            var crossValidationIndexSets = GetCrossValidationIndexSets(10, targets, sampler);
            var predictions = new double[observationCount];

            // Run cross validation loop.
            foreach (var set in crossValidationIndexSets)
            {
                // setup data.
                var trainingNameToData = new Dictionary<string, MemoryMinibatchData>
                {
                    { "observations", observationsData.GetSamples(set.training) },
                    { "targets", targetsData.GetSamples(set.training) }
                };

                var validationNameToData = new Dictionary<string, MemoryMinibatchData>
                {
                    { "observations", observationsData.GetSamples(set.validation) },
                    { "targets", targetsData.GetSamples(set.validation) }
                };

                var trainSource = new MemoryMinibatchSource(nameToVariable, trainingNameToData, seed: 232, randomize: true);
                var validationSource = new MemoryMinibatchSource(nameToVariable, validationNameToData, seed: 232, randomize: false);

                // Create model and fit.
                var model = CreateModel(inputVariable, targetVariable, targetCount, dataType, device);
                model.Fit(trainSource, batchSize: 8, epochs: 10);

                // Predict.
                var predictionsRaw = model.Predict(validationSource);
                var currentPredictions = predictionsRaw.Select(v => (double)v.Single()).ToArray();

                // set cross-validation predictions
                var validationIndices = set.validation;
                for (int i = 0; i < validationIndices.Length; i++)
                {
                    predictions[validationIndices[i]] = currentPredictions[i];
                }
            }

            Trace.WriteLine(FormatErrorString(targets, predictions));
        }

        static Model CreateModel(Function inputVariable, Variable targetVariable, int targetCount,
            DataType dataType, DeviceDescriptor device)
        {
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.GlorotNormal(random.Next());
            var biasInit = Initializers.Zero();

            // Create the architecture.
            var network = inputVariable

                .Dense(32, weightInit(), biasInit, device, dataType)
                .ReLU()
                .Dense(32, weightInit(), biasInit, device, dataType)
                .ReLU()
                .Dense(targetCount, weightInit(), biasInit, device, dataType);

            // loss
            var lossFunc = Losses.MeanSquaredError(network.Output, targetVariable);
            var metricFunc = Losses.MeanAbsoluteError(network.Output, targetVariable);

            // setup trainer.
            var learner = CntkCatalyst.Learners.Adam(network.Parameters());
            var trainer = CNTKLib.CreateTrainer(network, lossFunc, metricFunc, new LearnerVector { learner });

            var model = new Model(trainer, network, dataType, device);

            Trace.WriteLine(model.Summary());
            return model;
        }

        List<(int[] training, int[] validation)> GetCrossValidationIndexSets(int folds, double[] targets,
            IIndexSampler<double> sampler)
        {
            var samplesPerFold = targets.Length / folds;
            var allIndices = Enumerable.Range(0, targets.Length).ToArray();
            var currentIndices = Enumerable.Range(0, targets.Length).ToArray();

            var crossValidationIndexSets = new List<(int[] training, int[] validation)>();

            for (int i = 0; i < folds; i++)
            {
                var holdoutSample = sampler.Sample(targets, samplesPerFold, currentIndices);
                // Sample only from remaining indices.
                currentIndices = currentIndices.Except(holdoutSample).ToArray();
                // Training sample is all indices except the current hold out sample.
                var trainingSample = allIndices.Except(holdoutSample).ToArray();
                crossValidationIndexSets.Add((trainingSample, holdoutSample));
            }

            return crossValidationIndexSets;
        }

        string FormatErrorString(double[] targets, double[] predictions)
        {
            var metric = new MeanSquaredErrorRegressionMetric();
            var error = metric.Error(targets, predictions);
            return $"MSE: {error}";
        }
    }
}

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.Examples.Properties;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Optimization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Examples.Guides
{
    [TestClass]
    public class HyperparameterTuningGuide
    {
        [TestMethod]
        public void GradientBoost_Default_Parameters()
        {
            #region read and split data
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // creates training test splitter, 
            // Since this is a regression problem, we use the random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion

            // create learner with default parameters
            var learner = new RegressionSquareLossGradientBoostLearner(runParallel: false);

            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            TraceTrainingAndTestError(trainError, testError);
        }

        [TestMethod]
        public void GradientBoost_Optimize_Hyperparameters()
        {
            #region read and split data
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // creates training test splitter, 
            // Since this is a regression problem, we use the random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // Usually better results can be achieved by tuning a gradient boost learner

            var numberOfFeatures = trainSet.Observations.ColumnCount;

            // Parameter ranges for the optimizer
            // best parameter to tune on random forest is featuresPrSplit.
            var parameters = new double[][]
            {
                new double[] { 80, 300 }, // iterations (min: 20, max: 100)
                new double[] { 0.02, 0.2 }, // learning rate (min: 0.02, max: 0.2)
                new double[] { 8, 15 }, // maximumTreeDepth (min: 8, max: 15)
                new double[] { 0.5, 0.9 }, // subSampleRatio (min: 0.5, max: 0.9)
                new double[] { 1, numberOfFeatures }, // featuresPrSplit (min: 1, max: numberOfFeatures)
            };

            // Further split the training data to have a validation set to measure
            // how well the model generalizes to unseen data during the optimization.
            var validationSplit = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
                .SplitSet(trainSet.Observations, trainSet.Targets);


            // Define optimizer objective (function to minimize)
            Func<double[], OptimizerResult> minimize = p =>
            {
                // create the candidate learner using the current optimization parameters.
                var candidateLearner = new RegressionSquareLossGradientBoostLearner(
                        iterations: (int)p[0],
                        learningRate: p[1], 
                        maximumTreeDepth: (int)p[2], 
                        subSampleRatio: p[3], 
                        featuresPrSplit: (int)p[4],
                        runParallel: false);

                var candidateModel = candidateLearner.Learn(validationSplit.TrainingSet.Observations,
                    validationSplit.TrainingSet.Targets);

                var validationPredictions = candidateModel.Predict(validationSplit.TestSet.Observations);
                var candidateError = metric.Error(validationSplit.TestSet.Targets, validationPredictions);

                // trace current error 
                Trace.WriteLine(string.Format("Candidate Error: {0:0.0000}, Candidate Parameters: {1}",
                    candidateError, string.Join(", ", p)));

                return new OptimizerResult(p, candidateError);
            };

            // create random search optimizer
            var optimizer = new RandomSearchOptimizer(parameters, iterations: 30, runParallel: true);

            // find best hyperparameters
            var result = optimizer.OptimizeBest(minimize);
            var best = result.ParameterSet;

            // create the final learner using the best hyperparameters.
            var learner = new RegressionSquareLossGradientBoostLearner(
                iterations: (int)best[0],
                learningRate: best[1], 
                maximumTreeDepth: (int)best[2], 
                subSampleRatio: best[3],
                featuresPrSplit: (int)best[4], 
                runParallel: false);

            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // Optimizer found hyperparameters.
            Trace.WriteLine(string.Format("Found parameters, iterations:  {0}, learning rate {1:0.000}:  maximumTreeDepth: {2}, subSampleRatio {3:0.000}, featuresPrSplit: {4} ",
                (int)best[0], best[1], (int)best[2], best[3], (int)best[4]));
            TraceTrainingAndTestError(trainError, testError);
        }

        static void TraceTrainingAndTestError(double trainError, double testError)
        {
            Trace.WriteLine(string.Format("Train error: {0:0.0000} - Test error: {1:0.0000}",
                trainError, testError));
        }
    }
}

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Optimization;
using SharpLearning.RandomForest.Learners;
using System;
using System.Diagnostics;
using System.IO;

namespace SharpLearning.Examples.Guides
{
    [TestClass]
    public class MachineLearningGuide
    {
        [TestMethod]
        public void ReadAndSplitDataIntoTrainingTest_DecisionTree()
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

            // creates training test splitter, 
            // For a regression problem one would usually apply a random training/test set splitter.
            // However, since the quality target we are trying to predict does not contain floating point numbers
            // we are going to split the dataset using a stratified training/test set splitter.
            // this will make sure that we have an even distribution of the various quality grades in our training and test set.
            // 30 % of the data is used for the test set. 
            var splitter = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // 1. Start with something simple: default DecisionTreeLearner
            var learner = new RegressionDecisionTreeLearner();
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);
            
            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // fits the training set perfectly but high error on test set.
            TraceTrainingAndTestError(trainError, testError);

            // 2. Try adjusting the hyperparameters, depth is a good starting point for a decision tree
            learner = new RegressionDecisionTreeLearner(maximumTreeDepth: 10);
            model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            trainPredictions = model.Predict(trainSet.Observations);
            testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            trainError = metric.Error(trainSet.Targets, trainPredictions);
            testError = metric.Error(testSet.Targets, testPredictions);

            // higher error on training set but lower error on test set (this is better).
            TraceTrainingAndTestError(trainError, testError);
        }

        [TestMethod]
        public void DecisionTree_Optimize_Hyperparameters()
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
            // For a regression problem one would usually apply a random training/test set splitter.
            // However, since the quality target we are trying to predict does not contain floating point numbers
            // we are going to split the dataset using a stratified training/test set splitter.
            // this will make sure that we have an even distribution of the various quality grades in our training and test set.
            // 30 % of the data is used for the test set. 
            var splitter = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // 3. Use an optimizer for tuning hyperparameters

            // Parameter ranges for the optimizer 
            var parameters = new double[][]
            {
                new double[] { 1, 100 }, // maximumTreeDepth (min: 1, max: 100)
                new double[] { 1, 16 }, // minimumSplitSize (min: 1, max: 16)
            };

            // Define optimizer objective (function to minimize)
            Func<double[], OptimizerResult> minimize = p =>
            {
                // create the candidate learner using the current optimization parameters.
                var candidateLearner = new RegressionDecisionTreeLearner(maximumTreeDepth: (int)p[0], 
                    minimumSplitSize: (int)p[1]);

                // Further split the training data to have a validation set to measure
                // how well the model generalizes to unseen data during the optimization.
                var validationSplit = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
                    .SplitSet(trainSet.Observations, trainSet.Targets);

                var candidateModel = candidateLearner.Learn(validationSplit.TrainingSet.Observations,
                    validationSplit.TrainingSet.Targets);

                var validationPredictions = candidateModel.Predict(validationSplit.TestSet.Observations);
                var candidateError = metric.Error(validationSplit.TestSet.Targets, validationPredictions);

                // trace current error 
                Trace.WriteLine(string.Format("Candidate Error: {0:0.0000}, Candidate Parameters: {1}", 
                    candidateError, string.Join(", ", p)));

                return new OptimizerResult(p, candidateError);
            };

            // create optimizer
            var optimizer = new RandomSearchOptimizer(parameters, iterations: 30, runParallel: true);

            // find best hyperparameters
            var result = optimizer.OptimizeBest(minimize);
            var bestParameters = result.ParameterSet;

            // create learner with found parameters
            var learner = new RegressionDecisionTreeLearner(maximumTreeDepth: (int)bestParameters[0], 
                minimumSplitSize: (int)bestParameters[1]);

            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // Optimizer finds a much better set of hyperparameters.
            TraceTrainingAndTestError(trainError, testError);
        }

        [TestMethod]
        public void RandomForest_Default_Parameters()
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
            // For a regression problem one would usually apply a random training/test set splitter.
            // However, since the quality target we are trying to predict does not contain floating point numbers
            // we are going to split the dataset using a stratified training/test set splitter.
            // this will make sure that we have an even distribution of the various quality grades in our training and test set.
            // 30 % of the data is used for the test set. 
            var splitter = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // 4. More advanced learner RandomForest. Try defualt parameters first

            // create learner with found parameters
            var learner = new RegressionRandomForestLearner();
            
            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // With default parameters the random forest learner
            // already outperforms the optimized decision tree learner by alot.
            TraceTrainingAndTestError(trainError, testError);
        }

        [TestMethod]
        public void RandomForest_Optimize_Hyperparameters()
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
            // since this is a regression problem we use a random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // 5. Use an optimizer for tuning hyperparameters

            var numberOfFeatures = trainSet.Observations.ColumnCount;

            // Parameter ranges for the optimizer
            // best parameter to tune on random forest is featuresPrSplit.
            var parameters = new double[][]
            {
                new double[] { 100, 300 }, // trees (min: 30, max: 120)
                new double[] { 1, numberOfFeatures }, // featuresPrSplit (min: 0.5, max: 0.9)
                new double[] { 8, 100 }, // maximumTreeDepth (min: 8, max: 100)
                new double[] { 0.5, 0.9 }, // subSampleRatio (min: 0.5, max: 0.9)
            };

            // Define optimizer objective (function to minimize)
            Func<double[], OptimizerResult> minimize = p =>
            {
                // create the candidate learner using the current optimization parameters.
                var candidateLearner = new RegressionRandomForestLearner(trees: (int)p[0], featuresPrSplit: (int)p[1], 
                    maximumTreeDepth: (int)p[2], subSampleRatio: p[3],  runParallel: false);

                // Further split the training data to have a validation set to measure
                // how well the model generalizes to unseen data during the optimization.
                var validationSplit = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
                    .SplitSet(trainSet.Observations, trainSet.Targets);

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

            // create learner with found parameters
            var learner = new RegressionRandomForestLearner(trees: (int)best[0], featuresPrSplit: (int)best[1],
                maximumTreeDepth: (int)best[2], subSampleRatio: best[3]);

            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // Optimizer finds a much better set of hyperparameters.
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
            // For a regression problem one would usually apply a random training/test set splitter.
            // However, since the quality target we are trying to predict does not contain floating point numbers
            // we are going to split the dataset using a stratified training/test set splitter.
            // this will make sure that we have an even distribution of the various quality grades in our training and test set.
            // 30 % of the data is used for the test set. 
            var splitter = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // 6. Usually better results can be achieved by tuning a gradient boost learner

            var numberOfFeatures = trainSet.Observations.ColumnCount;

            // Parameter ranges for the optimizer
            // best parameter to tune on random forest is featuresPrSplit.
            var parameters = new double[][]
            {
                new double[] { 80, 100 }, // iterations (min: 20, max: 100)
                new double[] { 0.02, 0.2 }, // learning rate (min: 0.02, max: 0.2)
                new double[] { 8, 15 }, // maximumTreeDepth (min: 8, max: 15)
                new double[] { 0.5, 0.9 }, // subSampleRatio (min: 0.5, max: 0.9)
                new double[] { 1, numberOfFeatures }, // featuresPrSplit (min: 1, max: numberOfFeatures)
            };

            // Define optimizer objective (function to minimize)
            Func<double[], OptimizerResult> minimize = p =>
            {
                // create the candidate learner using the current optimization parameters.
                var candidateLearner = new RegressionSquareLossGradientBoostLearner(iterations: (int)p[0],
                learningRate: p[1], maximumTreeDepth: (int)p[2], subSampleRatio: p[3], featuresPrSplit: (int)p[4],
                runParallel: false);

                // Further split the training data to have a validation set to measure
                // how well the model generalizes to unseen data during the optimization.
                var validationSplit = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
                    .SplitSet(trainSet.Observations, trainSet.Targets);

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

            // create the candidate learner using the current optimization parameters.
            var learner = new RegressionSquareLossGradientBoostLearner(iterations: (int)best[0],
            learningRate: best[1], maximumTreeDepth: (int)best[2], subSampleRatio: best[3], 
            featuresPrSplit: (int)best[4], runParallel: false);

            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // Optimizer finds a much better set of hyperparameters.
            TraceTrainingAndTestError(trainError, testError);
        }

        static void TraceTrainingAndTestError(double trainError, double testError)
        {
            Trace.WriteLine(string.Format("Train error: {0:0.0000} - Test error: {1:0.0000}", trainError, testError));
        }
    }
}

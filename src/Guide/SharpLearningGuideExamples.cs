using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.InputOutput.Csv;
using SharpLearning.DecisionTrees.Learners;
using System.IO;
using SharpLearning.Metrics.Regression;
using SharpLearning.Examples.Properties;
using System.Diagnostics;
using SharpLearning.Optimization;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.RandomForest.Learners;
using SharpLearning.GradientBoost.Learners;

namespace SharpLearning.Examples.Guide
{
    [TestClass]
    public class SharpLearningGuideExamples
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
            // since this is a regression problem we use a random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

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
            learner = new RegressionDecisionTreeLearner(maximumTreeDepth: 5);
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

            // 3. Use an optimizer for tuning hyperparameters

            // Parameter ranges for the optimizer 
            var paramers = new double[][]
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

                // use cross validation to estimate model generalization on unseen data
                var cv = new RandomCrossValidation<double>(crossValidationFolds: 5, seed: 42);
                var predictions = cv.CrossValidate(candidateLearner, observations, targets);
                var candidateError = metric.Error(targets, predictions);

                // trace current error 
                Trace.WriteLine(string.Format("Candidate Error: {0:0.0000}, Candidate Parameters: {1}", 
                    candidateError, string.Join(", ", p)));

                return new OptimizerResult(p, candidateError);
            };

            // create random search optimizer
            //var optimizer = new RandomSearchOptimizer(paramers, iterations: 30, runParallel: true);
            //var optimizer = new ParticleSwarmOptimizer(paramers, 10, 3);
            var optimizer = new SequentialModelBasedOptimizer(paramers, 25, 5, 1);

            // find best hyperparameters
            var result = optimizer.OptimizeBest(minimize);
            var bestParameters = result.ParameterSet;

            // create learner with found parameters
            var learner = new RegressionDecisionTreeLearner(maximumTreeDepth: (int)bestParameters[0], 
                minimumSplitSize: (int)bestParameters[1]);

            // learn model with found parameters
            var model = learner.Learn(observations, targets);

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

            // 4. More advanced learner RandomForest. Try defualt parameters first

            // create learner with found parameters
            var learner = new RegressionRandomForestLearner();
            
            // learn model with found parameters
            var model = learner.Learn(observations, targets);

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
            var paramers = new double[][]
            {
                new double[] { 1, numberOfFeatures }, // featuresPrSplit (min: 1, max: numberOfFeatures)
                new double[] { 1, 16 }, // minimumSplitSize (min: 1, max: 16)
            };

            // Define optimizer objective (function to minimize)
            Func<double[], OptimizerResult> minimize = p =>
            {
                // create the candidate learner using the current optimization parameters.
                var candidateLearner = new RegressionRandomForestLearner(featuresPrSplit: (int)p[0],
                    minimumSplitSize: (int)p[1]);

                // use cross validation to estimate model generalization on unseen data
                var cv = new RandomCrossValidation<double>(crossValidationFolds: 5, seed: 42);
                var predictions = cv.CrossValidate(candidateLearner, observations, targets);
                var candidateError = metric.Error(targets, predictions);

                // trace current error 
                Trace.WriteLine(string.Format("Candidate Error: {0:0.0000}, Candidate Parameters: {1}",
                    candidateError, string.Join(", ", p)));

                return new OptimizerResult(p, candidateError);
            };

            // create random search optimizer
            //var optimizer = new RandomSearchOptimizer(paramers, iterations: 10, runParallel: false);
            //var optimizer = new ParticleSwarmOptimizer(paramers, 5, 2);
            var optimizer = new SequentialModelBasedOptimizer(paramers, 5, 5, 1);

            // find best hyperparameters
            var result = optimizer.OptimizeBest(minimize);
            var bestParameters = result.ParameterSet;

            // create learner with found parameters
            var learner = new RegressionRandomForestLearner(featuresPrSplit: (int)bestParameters[0],
                minimumSplitSize: (int)bestParameters[1]);

            // learn model with found parameters
            var model = learner.Learn(observations, targets);

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

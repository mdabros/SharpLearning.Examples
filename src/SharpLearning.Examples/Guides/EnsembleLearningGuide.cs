using System.Diagnostics;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.Examples.Guides
{
    [TestClass]
    public class EnsembleLearningGuide
    {
        [TestMethod]
        public void RegressionEnsembleLearner()
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

            // create the list of learners to include in the ensemble
            var ensembleLearners = new IIndexedLearner<double>[]
            {
                new RegressionAdaBoostLearner(maximumTreeDepth: 15),
                new RegressionRandomForestLearner(runParallel: false),
                new RegressionSquareLossGradientBoostLearner(iterations:  198, learningRate: 0.028,  maximumTreeDepth: 12,
                    subSampleRatio: 0.559, featuresPrSplit: 10, runParallel: false)

            };

            // create the ensemble learner
            var learner = new RegressionEnsembleLearner(learners: ensembleLearners);

            // the ensemble learnr combines all the provided learners
            // into a single ensemble model.
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

            // The ensemble model achieves a lower test error 
            // then any of the individual models:
            
            // RegressionAdaBoostLearner: 0.4005
            // RegressionRandomForestLearner: 0.4037
            // RegressionSquareLossGradientBoostLearner: 0.3936
            TraceTrainingAndTestError(trainError, testError);
        }


        static void TraceTrainingAndTestError(double trainError, double testError)
        {
            Trace.WriteLine(string.Format("Train error: {0:0.0000} - Test error: {1:0.0000}",
                trainError, testError));
        }

    }
}

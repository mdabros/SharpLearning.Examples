using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Diagnostics;
using System.IO;

namespace SharpLearning.Examples.CrossValidation
{
    [TestClass]
    public class TrainTestSplitExamples
    {
        [TestMethod]
        public void TrainingTestSplitter_SplitSet()
        {
            #region Read data
            
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix (all columns different from the targetName)
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            #endregion

            // creates training test splitter, observations are shuffled randomly
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainingSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;

            var learner = new RegressionDecisionTreeLearner();
            var model = learner.Learn(trainingSet.Observations, trainingSet.Targets);

            // predict test set
            var testPredictions = model.Predict(testSet.Observations);

            // metric for measuring model error
            var metric = new MeanSquaredErrorRegressionMetric();

            // The test set provides an estimate on how the model will perform on unseen data
            Trace.WriteLine("Test error: " + metric.Error(testSet.Targets, testPredictions));

            // predict training set for comparison
            var trainingPredictions = model.Predict(trainingSet.Observations);

            // The training set is NOT a good estimate of how well the model will perfrom on unseen data. 
            Trace.WriteLine("Training error: " + metric.Error(trainingSet.Targets, trainingPredictions));
        }
    }
}

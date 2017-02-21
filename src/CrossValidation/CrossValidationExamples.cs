using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;
using System.Diagnostics;
using System.IO;

namespace SharpLearning.Examples.CrossValidation
{
    [TestClass]
    public class CrossValidationExamples
    {
        [TestMethod]
        public void CrossValidation_CrossValidate()
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

            // creates cross validator, observations are shuffled randomly
            var cv = new RandomCrossValidation<double>(crossValidationFolds: 5, seed: 42);

            // create learner
            var learner = new RegressionDecisionTreeLearner();

            // cross-validated predictions
            var cvPredictions = cv.CrossValidate(learner, observations, targets);

            // metric for measuring model error
            var metric = new MeanSquaredErrorRegressionMetric();

            // cross-validation provides an estimate on how the model will perform on unseen data
            Trace.WriteLine("Cross-validation error: " + metric.Error(targets, cvPredictions));

            // train and predict training set for comparison. 
            var predictions = learner.Learn(observations, targets).Predict(observations);

            // The training set is NOT a good estimate of how well the model will perfrom on unseen data. 
            Trace.WriteLine("Training error: " + metric.Error(targets, predictions));
        }

        [TestMethod]
        public void CrossValidation_CrossValidate_ProbabilityPredictions()
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

            // creates cross validator, observations are shuffled randomly
            var cv = new RandomCrossValidation<ProbabilityPrediction>(crossValidationFolds: 5, seed: 42);

            // create learner
            var learner = new ClassificationDecisionTreeLearner();

            // cross-validated predictions
            var cvPredictions = cv.CrossValidate(learner, observations, targets);

            // metric for measuring model error
            var metric = new LogLossClassificationProbabilityMetric();

            // cross-validation provides an estimate on how the model will perform on unseen data
            Trace.WriteLine("Cross-validation error: " + metric.Error(targets, cvPredictions));

            // train and predict training set for comparison 
            var predictions = learner.Learn(observations, targets).PredictProbability(observations);

            // The training set is NOT a good estimate of how well the model will perfrom on unseen data. 
            Trace.WriteLine("Training error: " + metric.Error(targets, predictions));
        }
    }
}

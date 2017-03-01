using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Optimization;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using SharpLearning.InputOutput.Serialization;
using SharpLearning.Common.Interfaces;

namespace SharpLearning.Examples.Guides
{
    [TestClass]
    public class IntroductionGuide
    {

        [TestMethod]
        public void RandomForest_Default_Parameters()
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
            // Since this is a regression problem, we use the random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;

            // Create the learner and learn the model.
            var learner = new RegressionRandomForestLearner(trees: 100);
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
        public void RandomForest_Default_Parameters_Save_Load_Model_Using_Static_Methods()
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
            var learner = new RegressionRandomForestLearner(trees: 100);

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

            //Save model, in the file system use new StreamWriter(filePath);
            // default format is xml.
            var savedModel = new StringWriter();
            model.Save(() => savedModel);

            // load model, in the file system use new StreamReader(filePath);
            // default format is xml.
            var loadedModel = RegressionForestModel.Load(() => new StringReader(savedModel.ToString()));
        }

        [TestMethod]
        public void RandomForest_Default_Parameters_Save_Load_Model_Using_Serializer()
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
            var learner = new RegressionRandomForestLearner(trees: 100);

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

            //Save/load model as xml, in the file system use new StreamWriter(filePath);
            var xmlSerializer = new GenericXmlDataContractSerializer();
            var savedModelXml = new StringWriter();
            
            xmlSerializer
                .Serialize<IPredictorModel<double>>(model, () => savedModelXml);

            var loadedModelXml = xmlSerializer
                .Deserialize<IPredictorModel<double>>(() => new StringReader(savedModelXml.ToString()));

            //Save/load model as binary, in the file system use new StreamWriter(filePath);
            var binarySerializer = new GenericBinarySerializer();
            var savedModelBinary = new StringWriter();

            binarySerializer
                .Serialize<IPredictorModel<double>>(model, () => savedModelBinary);

            var loadedModelBinary = binarySerializer
                .Deserialize<IPredictorModel<double>>(() => new StringReader(savedModelBinary.ToString()));

        }

        [TestMethod]
        public void RandomForest_Default_Parameters_Variable_Importance()
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
            var learner = new RegressionRandomForestLearner(trees: 100);

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

            // the variable importance requires the featureNameToIndex
            // from the data set. This mapping describes the relation
            // from column name to index in the feature matrix.
            var featureNameToIndex = parser.EnumerateRows(c => c != targetName)
                .First().ColumnNameToIndex;

            // Get the variable importance from the model.
            // Variable importance is a measure made by to model 
            // of how important each feature is.
            var importances = model.GetVariableImportance(featureNameToIndex);

            // trace normalized importances as csv.
            var importanceCsv = new StringBuilder();
            importanceCsv.Append("FeatureName;Importance");
            foreach (var feature in importances)
            {
                importanceCsv.AppendLine();
                importanceCsv.Append(string.Format("{0};{1:0.00}",
                    feature.Key, feature.Value));
            }

            Trace.WriteLine(importanceCsv);
        }


        static void TraceTrainingAndTestError(double trainError, double testError)
        {
            Trace.WriteLine(string.Format("Train error: {0:0.0000} - Test error: {1:0.0000}", 
                trainError, testError));
        }
    }
}

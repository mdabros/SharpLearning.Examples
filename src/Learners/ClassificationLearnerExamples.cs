using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace SharpLearning.Examples.Learners
{
    [TestClass]
    public class ClassificationLearnerExamples
    {
        [TestMethod]
        public void ClassificationLearner_Learn()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();
            
            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new ClassificationDecisionTreeLearner();

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);
        }

        [TestMethod]
        public void ClassificationModel_Predict()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();
            
            // create learner
            var learner = new ClassificationDecisionTreeLearner();
            #endregion

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // predict all observations 
            var predictions = model.Predict(observations);

            // predict single observation
            var prediction = model.Predict(observations.Row(0));
        }

        [TestMethod]
        public void ClassificationModel_PredictProbability()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new ClassificationDecisionTreeLearner(maximumTreeDepth: 5);
            #endregion

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // predict probabilities for all observations 
            var probabilityPredictions = model.PredictProbability(observations);

            // predict probabilities for single observation
            var probabilityPrediction = model.PredictProbability(observations.Row(0));

            // the predicted class
            var predictedClass = probabilityPrediction.Prediction;

            // trace class probabilities
            probabilityPrediction.Probabilities.ToList()
                .ForEach(p => Trace.WriteLine(p.Key + ": " + p.Value));
        }

        [TestMethod]
        public void ClassificationModel_PredictProbability_Threshold_On_Probability()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets and convert to binary problem (low quality/high quality).
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector().Select(t => t < 5 ? 0.0 : 1.0).ToArray();

            var translation = new Dictionary<double, string> { { 0.0, "Low quality" }, { 1.0, "High quality" } }; 

            // create learner
            var learner = new ClassificationDecisionTreeLearner(maximumTreeDepth: 5);
            #endregion

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // predict probabilities for all observations 
            var probabilityPredictions = model.PredictProbability(observations);

            // zip target and probabilities to keep order
            var zip = targets.Zip(probabilityPredictions, (t, p) => new { Target = t, Prediction = p });

            // threhold on the probabilty of the predicted class.
            // This will remove the obserations that the model is uncertain about.
            var probabilityThreshold = 0.90;
            var thresholdedResult = zip.Where(kvp => kvp.Prediction.Probabilities[kvp.Prediction.Prediction] > probabilityThreshold);
            
            // evaluate the resulting observations
            var thresholdedPredictions = thresholdedResult.Select(p => p.Prediction).ToArray();
            var thresholdedTargets = thresholdedResult.Select(p => p.Target).ToArray();

            // evaluate only on probability thresholded data
            var metric = new LogLossClassificationProbabilityMetric();
            Trace.WriteLine("ProbabilityThresholded Result:");
            Trace.WriteLine(metric.ErrorString(thresholdedTargets, thresholdedPredictions, translation));
            Trace.WriteLine("");

            // evaluate on all data for comparison
            Trace.WriteLine("All data result:");
            Trace.WriteLine(metric.ErrorString(targets, probabilityPredictions, translation));
        }

        [TestMethod]
        public void ClassificationModel_FeatureImportance()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new ClassificationDecisionTreeLearner();

            #endregion

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // raw feature importance
            var rawImportance = model.GetRawVariableImportance();

            // Normalized and named feature importance.
            // This gives information about which features/variables the learner found important (higher is more important).
            var featureNameToIndex = parser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex;
            var importance = model.GetVariableImportance(featureNameToIndex);

            // trace normalized importances
            var importanceCsv = new StringBuilder();
            importanceCsv.Append("FeatureName;Importance");
            foreach (var feature in importance)
            {
                importanceCsv.AppendLine();
                importanceCsv.Append(feature.Key + ";" + feature.Value);
            }

            Trace.WriteLine(importanceCsv);
        }

        [TestMethod]
        public void ClassificationModel_Save_Load()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new ClassificationDecisionTreeLearner();

            #endregion

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);

            var writer = new StringWriter();
            model.Save(() => writer);

            // save to file 
            //model.Save(() => new StreamWriter(filePath));

            var text = writer.ToString();
            var loadedModel = ClassificationDecisionTreeModel.Load(() => new StringReader(text));
            
            // load from file 
            //ClassificationDecisionTreeModel.Load(() => new StreamReader(filePath));
        }
    }
}

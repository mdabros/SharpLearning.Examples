using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.RandomForest.Learners;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace SharpLearning.Examples.Learners
{
    [TestClass]
    public class RegressionLearnerExamples
    {
        [TestMethod]
        public void RegressionLearner_Learn()
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

            // create learner
            var learner = new RegressionDecisionTreeLearner();

            // learns a RegressionDecisionTreeModel
            var model = learner.Learn(observations, targets);
        }

        [TestMethod]
        public void RegressionModel_Predict()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new RegressionDecisionTreeLearner();
            #endregion

            // learns a RegressionDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // predict all observations 
            var predictions = model.Predict(observations);

            // predict single observation
            var prediction = model.Predict(observations.Row(0));
        }

        [TestMethod]
        public void RegressionModel_FeatureImportance()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new RegressionDecisionTreeLearner();
            #endregion

            // learns a RegressionDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // raw feature importance
            var rawImportance = model.GetRawVariableImportance();

            // normalized and named feature importance
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
        public void RegressionModel_Save_Load()
        {
            #region learner creation

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner
            var learner = new RegressionDecisionTreeLearner();

            #endregion

            // learns a ClassificationDecisionTreeModel
            var model = learner.Learn(observations, targets);

            var writer = new StringWriter();
            model.Save(() => writer);

            // save to file 
            //model.Save(() => new StreamWriter(filePath));

            var text = writer.ToString();
            var loadedModel = RegressionDecisionTreeModel.Load(() => new StringReader(text));

            // load from file 
            //RegressionDecisionTreeModel.Load(() => new StreamReader(filePath));
        }
    }
}

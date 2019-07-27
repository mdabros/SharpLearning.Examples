using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Examples.LearningCurves
{
    [TestClass]
    public class LearningCurvesExamples
    {
        [TestMethod]
        public void LearningCurves_Calculate()
        {
            #region Read data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            #endregion

            // metric for measuring model error
            var metric = new MeanSquaredErrorRegressionMetric();

            // creates cross validator, observations are shuffled randomly
            var learningCurveCalculator = new RandomShuffleLearningCurvesCalculator<double>(metric, 
                samplePercentages: new double[] { 0.05, 0.1, 0.2, 0.4, 0.8, 1.0 },
                trainingPercentage: 0.7, numberOfShufflesPrSample: 5);

            // create learner
            var learner = new RegressionDecisionTreeLearner(maximumTreeDepth: 5);

            // calculate learning curve
            var learningCurve = learningCurveCalculator.Calculate(learner, observations, targets);
            
            // write to csv
            var writer = new StringWriter();
            learningCurve.Write(() => writer);

            // trace result
            // Plotting the learning curves will help determine if the model has high bias or high variance.
            // This information can be used to determine what to try next in order to improve the model. 
            Trace.WriteLine(writer.ToString());

            // alternatively, write to file
            //learningCurve.Write(() => new StreamWriter(filePath));
        }

        [TestMethod]
        public void LearningCurves_Calculate_ProbabilityPrediction()
        {
            #region Read data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets and convert to binary problem (low quality/high quality).
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector().Select(t => t < 5 ? 0.0 : 1.0).ToArray();

            #endregion

            // metric for measuring model error
            var metric = new LogLossClassificationProbabilityMetric();

            // creates cross validator, observations are shuffled randomly
            var learningCurveCalculator = new RandomShuffleLearningCurvesCalculator<ProbabilityPrediction>(metric,
                samplePercentages: new double[] { 0.05, 0.1, 0.2, 0.4, 0.8, 1.0 },
                trainingPercentage: 0.7, numberOfShufflesPrSample: 5);

            // create learner
            var learner = new ClassificationDecisionTreeLearner(maximumTreeDepth: 5);

            // calculate learning curve
            var learningCurve = learningCurveCalculator.Calculate(learner, observations, targets);

            // write to csv
            var writer = new StringWriter();
            learningCurve.Write(() => writer);

            // trace result 
            // Plotting the learning curves will help determine if the model has high bias or high variance.
            // This information can be used to determine what to try next in order to improve the model. 
            Trace.WriteLine(writer.ToString());

            // alternatively, write to file
            //learningCurve.Write(() => new StreamWriter(filePath));
        }
    }
}

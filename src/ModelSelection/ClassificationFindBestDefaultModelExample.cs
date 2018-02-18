using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.Examples.ModelSelection
{
    [TestClass]
    public class ClassificationFindBestDefaultModelExample
    {
        [TestMethod]
        public void Classification_Find_Best_Model_With_Default_Parameters()
        {
            #region Read and Transform Data
              var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix (all columns different from the targetName)
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // create minmax normalizer (normalizes each feature from 0.0 to 1.0)
            var minMaxTransformer = new MinMaxTransformer(0.0, 1.0);

            // transforms features using the feature normalization transform 
            minMaxTransformer.Transform(observations, observations);

            // read targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();
            #endregion

            // split data
            // creates training test splitter, training and test set are splittet
            // to have equal distribution of classes in both set.
            var splitter = new StratifiedTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainingSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;

            // Create list of all classification learners (with default parameters)
            var learners = new List<ILearner<double>>
            {
                new ClassificationDecisionTreeLearner(),
                new ClassificationRandomForestLearner(),
                new ClassificationExtremelyRandomizedTreesLearner(),
                new ClassificationAdaBoostLearner(),
                new ClassificationBinomialGradientBoostLearner(),
            };

            // metric for measuring the error
            var metric = new TotalErrorClassificationMetric<double>();

            // try all learners
            var testPredictions = new double[testSet.Targets.Length];
            var testObservation = new double[trainingSet.Observations.ColumnCount];
            foreach (var learner in learners)
            {
                // train model
                var model = learner.Learn(trainingSet.Observations, trainingSet.Targets);

                // iterate over test set and predict each observation
                for (int i = 0; i < testSet.Targets.Length; i++)
                {
                    testSet.Observations.Row(i, testObservation);
                    testPredictions[i] = model.Predict(testObservation);
                }
                
                // measure error on test set
                var error = metric.Error(testSet.Targets, testPredictions);

                // Trace learner type and error to output window
                Trace.WriteLine(string.Format("{0}: {1:0.0000}", learner.GetType().Name, error));
            }
        }
    }
}

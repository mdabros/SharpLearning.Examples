using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Optimization;
using System;
using System.Diagnostics;
using System.IO;

namespace SharpLearning.Examples.Optimization
{
    [TestClass]
    public class HyperparameterTuningExamples
    {
        [TestMethod]
        public void GridSearch_Parameter_Tuning()
        {
            #region Read data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            #endregion

            // metric to minimize
            var metric = new MeanSquaredErrorRegressionMetric();

            // parameters to try in grid search
            var paramers = new double[][]
            {
                new double[] { 1, 100 }, // maximumTreeDepth (min: 1, max: 100)
                new double[] { 1, 16 }, // minimumSplitSize (min: 1, max: 16)
            };

            // create random search optimizer
            var optimizer = new RandomSearchOptimizer(paramers, iterations: 30, runParallel: true);
            
            // other availible optimizers
            // GridSearchOptimizer
            // GlobalizedBoundedNelderMeadOptimizer
            // ParticleSwarmOptimizer
            // SequentialModelBasedOptimizer

            // function to minimize
            Func<double[], OptimizerResult> minimize = p =>
            {
                var cv = new RandomCrossValidation<double>(crossValidationFolds: 5, seed: 42);
                var optlearner = new RegressionDecisionTreeLearner(maximumTreeDepth: (int)p[0], minimumSplitSize: (int)p[1]);
                var predictions = cv.CrossValidate(optlearner, observations, targets);
                var error = metric.Error(targets, predictions);
                Trace.WriteLine("Error: " + error);
                return new OptimizerResult(p, error);
            };

            // run optimizer
            var result = optimizer.OptimizeBest(minimize);
            var bestParameters = result.ParameterSet;

            Trace.WriteLine("Result: " + result.Error);

            // create learner with found parameters
            var learner = new RegressionDecisionTreeLearner(maximumTreeDepth: (int)bestParameters[0], minimumSplitSize: (int)bestParameters[1]);
           
            // learn model with found parameters
            var model = learner.Learn(observations, targets);
        }
    }
}

using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Examples.Metrics
{
    [TestClass]
    public class RegressionMetricExamples
    {
        [TestMethod]
        public void MeanSquaredErrorRegressionMetric_Error()
        {
            var targets = new double[] { 1, 2, 2, 2, 3, 1, 1, 2, 3 };
            var predictions = new double[] { 1, 2, 2, 2, 1, 2, 2, 1, 3 };

            var metric = new MeanSquaredErrorRegressionMetric();
            Trace.WriteLine("Error: " + metric.Error(targets, predictions));
        }
    }
}

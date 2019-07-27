using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Examples.Metrics
{
    [TestClass]
    public class ClassificationMetricExamples
    {
        [TestMethod]
        public void ClassificationMetric_Error()
        {
            var targets = new double[] { 1, 2, 2, 2, 3, 1, 1, 2, 3 };
            var predictions = new double[] { 1, 2, 2, 2, 1, 2, 2, 1, 3 };

            var metric = new TotalErrorClassificationMetric<double>();
            
            Trace.WriteLine("Error: " + metric.Error(targets, predictions));
        }

        [TestMethod]
        public void ClassificationMetric_ErrorString()
        {
            var targets = new double[] { 1, 2, 2, 2, 3, 1, 1, 2, 3 };
            var predictions = new double[] { 1, 2, 2, 2, 1, 2, 2, 1, 3 };

            var metric = new TotalErrorClassificationMetric<double>();
            
            Trace.WriteLine(metric.ErrorString(targets, predictions));
        }

        [TestMethod]
        public void ClassificationMetric_ErrorString_Translate_Target_Values_To_Names()
        {
            var targets = new double[] { 1, 2, 2, 2, 3, 1, 1, 2, 3 };
            var predictions = new double[] { 1, 2, 2, 2, 1, 2, 2, 1, 3 };

            var translation = new Dictionary<double, string> { { 1.0, "Quality1" }, { 2.0, "Quality2" }, { 3.0, "Quality3" } };
            var metric = new TotalErrorClassificationMetric<double>();
            
            Trace.WriteLine(metric.ErrorString(targets, predictions, translation));
        }

        [TestMethod]
        public void ClassificationMetric_On_Strings()
        {
            var targets = new string[] { "Quality1", "Quality2", "Quality2", "Quality2", "Quality3", "Quality1", "Quality1", "Quality2", "Quality3" };
            var predictions = new string[] { "Quality1", "Quality2", "Quality2", "Quality2", "Quality1", "Quality2", "Quality2", "Quality1", "Quality3" };

            var metric = new TotalErrorClassificationMetric<string>();
            Trace.WriteLine(metric.ErrorString(targets, predictions));
        }
    }
}

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Examples.Properties;
using SharpLearning.InputOutput.Csv;
using System.IO;

namespace SharpLearning.Examples.InputOutput
{
    [TestClass]
    public class CsvParserExamples
    {
        [TestMethod]
        public void CsvParser_Read_F64Matrix()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));
            
            // Read numeric feature matrix using conditional expression (exclude colum with header value quality)
            var obserationsConditionals = parser.EnumerateRows(c => c != "quality")
                .ToF64Matrix();

            // Read numeric feature matrix using specified header values
            var obserationsSpecified = parser.EnumerateRows("volatile acidity", "citric acid", "residual sugar")
                .ToF64Matrix();
        }

        [TestMethod]
        public void CsvParser_Read_F64Vector()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));

            // Read double array using specified header values
            var targets = parser.EnumerateRows("quality")
                .ToF64Vector();
        }

        [TestMethod]
        public void CsvParser_Read_StringMatrix()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));

            // Read feature matrix as strings using conditional expression (exclude colum with header value quality)
            var obserationsConditionals = parser.EnumerateRows(c => c != "quality")
                .ToStringMatrix();

            // Read feature matrix as strings using specified header values
            var obserationsSpecified = parser.EnumerateRows("volatile acidity", "citric acid", "residual sugar")
                .ToStringMatrix();
        }

        [TestMethod]
        public void CsvParser_Read_StringVector()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.winequality_white));

            // Read string array using specified header values
            var targets = parser.EnumerateRows("quality")
                .ToStringVector();
        }
    }
}

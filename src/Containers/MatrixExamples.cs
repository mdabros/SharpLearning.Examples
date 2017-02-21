using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Examples.Containers
{
    [TestClass]
    public class MatrixExamples
    {
        [TestMethod]
        public void F64Matrix_Create()
        {
            // matrix is created row wise
            var matrix = new F64Matrix(new double[] { 1, 2, 3, 
                                                      4, 5, 6 }, 2, 3);

            // dimensions
            Assert.AreEqual(2, matrix.RowCount);
            Assert.AreEqual(3, matrix.ColumnCount);
        }

        [TestMethod]
        public void F64Matrix_GetRow()
        {
            // matrix is created row wise
            var matrix = new F64Matrix(new double[] { 1, 2, 3, 
                                                      4, 5, 6 }, 2, 3);
            // returns the row as an array
            var row = matrix.Row(1); // [4, 5, 6]
        }

        [TestMethod]
        public void F64Matrix_GetColumn()
        {
            // matrix is created row wise
            var matrix = new F64Matrix(new double[] { 1, 2, 3, 
                                                      4, 5, 6 }, 2, 3);
            // returns the column as an array
            var column = matrix.Column(1); // [2, 5]
        }

        [TestMethod]
        public void F64Matrix_GetRows()
        {
            // matrix is created row wise
            var matrix = new F64Matrix(new double[] { 1, 2, 3, 
                                                      4, 5, 6,
                                                      7, 8, 9}, 3, 3);
            
            // returns selected rows a a new matrix
            var rows = matrix.Rows(new int[] { 0, 2 }); // [1, 2, 3,
                                                          //  7, 8, 9]
        }

        [TestMethod]
        public void F64Matrix_GetColumns()
        {
            // matrix is created row wise
            var matrix = new F64Matrix(new double[] { 1, 2, 3, 
                                                      4, 5, 6,
                                                      7, 8, 9}, 3, 3);

            // returns selected columns a a new matrix
            var cols = matrix.Columns(new int[] { 0, 2 }); // [1, 3,
                                                             //  4, 6
                                                             //  7, 9]
        }
    }
}

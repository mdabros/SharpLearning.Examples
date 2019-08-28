using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DataSource;
using SharpLearning.InputOutput.Csv;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TorchSharp;
using TorchSharp.Tensor;
using static TorchSharp.NN.LossFunction;

namespace SharpLearning.Examples.IntegrationWithOtherMLPackages
{
    [TestClass]
    public class TorchSharp_DataSource
    {
        const string TargetsId = nameof(TargetsId);
        const string ImagesId = nameof(ImagesId);
        const int _logInterval = 10;

        [TestMethod]
        public void TorchSharp_With_SharpLearning_DataSource()
        {
            Torch.SetSeed(1);

            var epochs = 10;
            var trainBatchSize = 32;
            var trainSize = 60000;

            var testBatchSize = 1;
            var testSize = 20000;

            var trainingIdToDataLoader = CreateImageDataLoaders(@"E:\DataSets\CIFAR10\train_map.csv", @"E:\DataSets\CIFAR10\train");
            var trainingSource = new DataSource<float>(trainingIdToDataLoader, batchSize: trainBatchSize, shuffle: true, seed: 232);

            var testIdToDataLoader = CreateImageDataLoaders(@"E:\DataSets\CIFAR10\test_map.csv", @"E:\DataSets\CIFAR10\test");
            var testSource = new DataSource<float>(testIdToDataLoader, batchSize: testBatchSize, shuffle: false, seed: 232);;
                                    
            using (var model = new Model(classCount: 10))
            using (var optimizer = TorchSharp.NN.Optimizer.Adam(model.Parameters(), 0.001))
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int epoch = 0; epoch < epochs;)
                {
                    var (minibatch, isSweepEnd) = trainingSource.GetNextBatch();

                    var getImages = GetTensor(minibatch[ImagesId]);
                    var getTargets = GetTensor(minibatch[ImagesId]);

                    using (var images = getImages())
                    using (var targets = getTargets())
                    {
                        Train(model, optimizer, NLL(), images, targets, 
                            epoch, trainBatchSize, trainSize);

                        //Test(model, NLL(), test, testSize);

                        if (isSweepEnd)
                        {
                            epoch++;
                        }
                    }
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time {sw.ElapsedMilliseconds}.");
                Console.ReadLine();
            }
        }

        static Dictionary<string, DataLoader<float>> CreateImageDataLoaders(string dataFilePath, string imagesDirectoryPath)
        {
            // targets data loader.
            var valueToOneHotIndex = Enumerable.Range(0, 10).ToDictionary(v => (float)v, v => v);
            var targetsLoader = CsvParser.FromFile(dataFilePath, separator: '\t')
                .EnumerateRows("target").Select(v => v.Values)
                .ToCsvDataLoader(columnParser: s => ToOneHot(float.Parse(s), valueToOneHotIndex), 
                    sampleShape: valueToOneHotIndex.Count);

            // enumerate images.
            var imageFilePaths = CsvParser.FromFile(dataFilePath, separator: '\t')
                .EnumerateRows("filepath").ToStringVector()
                .Select(filename => Path.Combine(imagesDirectoryPath, filename));

            // images data loader.
            var random = new Random(Seed: 232);
            var sampleShape = new[] { 32, 32, 3 };
            var imagesLoader = DataLoaders.EnumerateImages<Rgba32>(imageFilePaths)               
                .Select(i =>
                // Add augmentations.
                    i.Flip(FlipMode.Horizontal, random)
                    .Zoom(maxZoom: 1.2f, random)
                    .Brightness(0.9f, 1.1f, random))
                // Create data loader.
                .ToImageDataLoader(pixelConverter: Converters.ToFloat, sampleShape: sampleShape);

            var idToLoader = new Dictionary<string, DataLoader<float>>
            {
                { ImagesId, imagesLoader },
                { TargetsId, targetsLoader },
            };

            return idToLoader;
        }

        Func<TorchTensor> GetTensor<T>(DataBatch<T> data)
        {
            var sampleCount = data.SampleCount;
            var sampleShape = data.SampleShape;
            var longShape = new long[sampleShape.Length + 1];
            longShape[0] = sampleCount;
            for (int i = 0; i < sampleShape.Length; i++)
            {
                longShape[i + 1] = sampleShape[i];
            }

            return () => data.Data.ToTorchTensor(longShape);
        }

        /// <summary>
        /// Transform value to onehot vector.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="valueToOneHotIndex"></param>
        /// <returns></returns>
        public static float[] ToOneHot(float value, IReadOnlyDictionary<float, int> valueToOneHotIndex)
        {
            var oneHotVector = new float[valueToOneHotIndex.Count];
            var oneHotIndex = valueToOneHotIndex[value];
            oneHotVector[oneHotIndex] = 1;
            return oneHotVector;
        }

        private class Model : TorchSharp.NN.Module
        {
            private readonly TorchSharp.NN.Module features;
            private readonly TorchSharp.NN.Module avgPool;
            private readonly TorchSharp.NN.Module classifier;

            public Model(int classCount)
            {
                features = Sequential(
                    Conv2D(3, 64, kernelSize: 3, stride: 2, padding: 1),
                    Relu(inPlace: true),
                    MaxPool2D(kernelSize: new long[] { 2 }),
                    Conv2D(64, 192, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    MaxPool2D(kernelSize: new long[] { 2 }),
                    Conv2D(192, 384, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    Conv2D(384, 256, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    Conv2D(256, 256, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    MaxPool2D(kernelSize: new long[] { 2 }));

                avgPool = AdaptiveAvgPool2D(2, 2);

                classifier = Sequential(
                    Dropout(IsTraining()),
                    Linear(256 * 2 * 2, 4096),
                    Relu(inPlace: true),
                    Dropout(IsTraining()),
                    Linear(4096, 4096),
                    Relu(inPlace: true),
                    Linear(4096, classCount)
                );

                RegisterModule(features);
                RegisterModule(classifier);
            }

            public override TorchTensor Forward(TorchTensor input)
            {
                using (var f = features.Forward(input))
                using (var avg = avgPool.Forward(f))

                using (var x = avg.View(new long[] { avg.Shape[0], 256 * 2 * 2 }))
                    return classifier.Forward(x);
            }
        }

        private static void Train(
            TorchSharp.NN.Module model,
            TorchSharp.NN.Optimizer optimizer,
            Loss loss,
            TorchTensor data, 
            TorchTensor target,
            int epoch,
            long batchSize,
            long size)
        {
            model.Train();

            int batchId = 1;
            long total = 0;
            long correct = 0;

            optimizer.ZeroGrad();

            using (var prediction = model.Forward(data))
            using (var output = loss(TorchSharp.NN.Module.LogSoftMax(prediction, 1), target))
            {
                output.Backward();

                optimizer.Step();

                var predicted = prediction.Argmax(1);
                total += target.Shape[0];
                correct += predicted.Eq(target).Sum().DataItem<long>();

                if (batchId % _logInterval == 0)
                {
                    Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.DataItem<float>()} Acc: { (float)correct / total }");
                }

                batchId++;

                data.Dispose();
                target.Dispose();
            }
        }

        private static void Test(
            TorchSharp.NN.Module model,
            Loss loss,
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
            long size)
        {
            model.Eval();

            double testLoss = 0;
            long correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var prediction = model.Forward(data))
                using (var output = loss(TorchSharp.NN.Module.LogSoftMax(prediction, 1), target))
                {
                    testLoss += output.DataItem<float>();

                    var pred = prediction.Argmax(1);

                    correct += pred.Eq(target).Sum().DataItem<long>();

                    data.Dispose();
                    target.Dispose();
                    pred.Dispose();
                }

            }

            Console.WriteLine($"\rTest set: Average loss {testLoss} | Accuracy {(float)correct / size}");
        }
    }
}

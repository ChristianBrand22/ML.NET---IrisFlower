using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace IrisFlower
{
    /// <summary>
    /// A data transfer class that holds a single iris flower.
    /// </summary>
    public class IrisData
    {
        [LoadColumn(0)] public float SepalLength;
        [LoadColumn(1)] public float SepalWidth;
        [LoadColumn(2)] public float PetalLength;
        [LoadColumn(3)] public float PetalWidth;
        [LoadColumn(4)] public string Label;
    }

    /// <summary>
    /// A prediction class that holds a single model prediction.
    /// </summary>
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint ClusterID;

        [ColumnName("Score")]
        public float[] Score;
    }

    // the rest of the code goes here....

    // The main program class.
    class Program
    {
        // The program entry point.
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // read the iris flower data from a text file
            var data = mlContext.Data.LoadFromTextFile<IrisData>(
                path: "iris-data.csv",
                hasHeader: false,
                separatorChar: ',');

            // split the data into a training and testing partition
            var partitions = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // the rest of the code goes here...

            // set up a learning pipeline
            // step 1: concatenate features into a single column
            var pipeline = mlContext.Transforms.Concatenate(
                    "Features",
                    "SepalLength", 
                    "PetalLength",
                    "PetalWidth")

                // step 2: use k-means clustering to find the flower species
                .Append(mlContext.Clustering.Trainers.KMeans(
                    featureColumnName: "Features",
                    numberOfClusters: 3));

            // train the model on the data file
            var model = pipeline.Fit(partitions.TrainSet);

            // the rest of the code goes here...

            // evaluate the model
            Console.WriteLine("Evaluating model:");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = mlContext.Clustering.Evaluate(
                predictions,
                scoreColumnName: "Score",
                featureColumnName: "Features");
            Console.WriteLine($"   Average distance:     {metrics.AverageDistance}");
            Console.WriteLine($"   Davies-Bouldin index: {metrics.DaviesBouldinIndex}");

            // the rest of the code goes here....

            // show predictions for a couple of flowers
            Console.WriteLine("Predicting 3 flowers from the test set....");
            var flowers = mlContext.Data.CreateEnumerable<IrisData>(partitions.TestSet, reuseRowObject: false).ToArray();
            var flowerPredictions = mlContext.Data.CreateEnumerable<IrisPrediction>(predictions, reuseRowObject: false).ToArray();
            foreach (var i in new int[] { 0, 10, 20 })
            {
                Console.WriteLine($"   Flower: {flowers[i].Label}, prediction: {flowerPredictions[i].ClusterID}");
            }
        }
    }


}
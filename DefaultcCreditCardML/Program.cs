using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace DefaultcCreditCardML
{
    class Program
    {

        static string dataPath = "creditcard.csv";
        static void Main(string[] args)
        {
            // MLCreditCard();
            MLCreditCard();
        }

        private static void MLCreditCard()
        {
            Console.WriteLine("Different Credit Card Algorithms evaluation");

            string[] InputColumns;
            MLContext ctx;
            DataOperationsCatalog.TrainTestData splitData;
            LoadandSplitData(out InputColumns, out ctx, out splitData);


            //create single features column
            var concatenatingEstimator = ctx.Transforms.Concatenate("Features", InputColumns);


            //select algorithm
            var estimator = concatenatingEstimator.Append(ctx.BinaryClassification.Trainers.FastForest("Label", "Features"));

            //train model
            var model = estimator.Fit(splitData.TrainSet);

            var predictions = model.Transform(splitData.TestSet);

            var metrics = ctx.BinaryClassification.EvaluateNonCalibrated(predictions, "Label");


            Console.WriteLine($"Accuracy : {metrics.Accuracy.ToString("0.00")} + F1 Score: {metrics.F1Score.ToString("0.00")} ");
            Console.WriteLine($"Negative Precision : {metrics.NegativePrecision.ToString("0.00")} + Negative Recall : {metrics.NegativeRecall.ToString("0.00")}");
            Console.WriteLine($"Positive Precision :{metrics.PositivePrecision.ToString("0.00")} + Positive Recall : {metrics.NegativePrecision.ToString("0.00")}");
            Console.WriteLine($"AUPRC :{metrics.AreaUnderPrecisionRecallCurve.ToString("0.00")} + AURC : {metrics.AreaUnderRocCurve.ToString("0.00")}");
            
            Console.Read();
            //  Ml();
        }

        private static void LoadandSplitData(out string[] InputColumns, out MLContext ctx, out DataOperationsCatalog.TrainTestData splitData)
        {
            InputColumns = (new CreditCardData()).GetType().GetFields().Where(x => x.FieldType != typeof(bool)).Select(x => x.Name).ToArray();


            //Step 1. Create a ML Context
            ctx = new MLContext();

            //Step 2. Read in the input data for model training
            IDataView dataReader = ctx.Data
                .LoadFromTextFile<CreditCardData>(dataPath, separatorChar: ',', hasHeader: true);


            //split data
            splitData = ctx.Data.TrainTestSplit(dataReader, 0.2);
        }

      public static  void CHeckFeatures(MLContext context, IDataView data)// EstimatorChain<ITransformer> pp)// pipeline)//EstimatorChain<BinaryPredictionTransformer<LinearBinaryModelParameters>> pipeline)//IEstimator<ITransformer> pipeline)
        {
            string[] InputColumns;
            DataOperationsCatalog.TrainTestData splitData;
            LoadandSplitData(out InputColumns, out context, out splitData);
            var pipeline = context.Transforms.Concatenate("Features",InputColumns)
                                   .Append(context.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features"));
      
            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            var linearModel = model.LastTransformer;

            var featureContributionCalculation = context.Transforms.CalculateFeatureContribution(linearModel, normalize: false);

            var featureContributionData = featureContributionCalculation.Fit(transformedData).Transform(transformedData);
            var featureContributionDataPreview = featureContributionCalculation.Fit(transformedData).Transform(transformedData).Preview();

            var shuffledSubset = context.Data.TakeRows(context.Data.ShuffleRows(featureContributionData), 10);
            var scoringEnumerator = context.Data.CreateEnumerable<CreditCardData>(shuffledSubset, true);

            foreach (var row in scoringEnumerator)
            {
                Console.WriteLine(row);
            }

            Console.Read();
        }

    }
}

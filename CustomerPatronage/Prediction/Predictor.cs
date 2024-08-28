using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CustomerPatronage.Persistance;
using CustomerPatronage.Training;
using Microsoft.ML;

namespace CustomerPatronage.Prediction
{
    public class Predictor
    {

        private MLContext mlContext = new MLContext();

        private ITransformer LoadModel(
            string modelName,
            int predictionMonthsWindow
        )
        {
            var modelPath = new FileManager()
                .GetModelPath(
                    modelName: modelName,
                    predictionMonthsWindow: predictionMonthsWindow
                );
            // Load the model from disk
            if (File.Exists(modelPath))
            {
                var model = mlContext.Model.Load(modelPath, out var modelInputSchema);
                return model;
            }
            else
            {
                throw new FileNotFoundException("Model file not found", modelPath);
            }
        }
        private PurchasePredictionOutput PredictSingleCustomer(
            PurchasePredictionInput input,
            string modelName,
            int predictionMonthsWindow)
        {

            // Calculate baseline averages for fallback
            var averageSpend = data.Average(d => d.Spend);
            var averageFrequency = data.GroupBy(d => d.CustomerId).Average(g => g.Count() / (float)predictionMonthsWindow);

            var model = LoadModel(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PurchasePredictionInput, PurchasePredictionOutput>(model);
            var prediction = predictionEngine.Predict(input);

            // If prediction output is below a certain threshold, we may assume this is a cold start case
            if (prediction.ExpectedSpend < 0.01f && prediction.PurchaseFrequency < 0.01f)
            {
                // Return baseline prediction with the flag set
                return new PurchasePredictionOutput
                {
                    CustomerId = input.CustomerId,
                    CustomerName = input.CustomerName,
                    ExpectedSpend = averageSpend,
                    PurchaseFrequency = averageFrequency,
                    IsBaseline = true // This indicates it's a baseline prediction
                };
            }

            // Normal prediction path
            prediction.IsBaseline = false; // This is a regular prediction
            return prediction;
        }
    }
}
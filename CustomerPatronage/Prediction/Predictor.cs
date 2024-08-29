using System.Text.Json;
using CustomerPatronage.Persistance;
using CustomerPatronage.Training;
using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;
using System;
using System.IO;

namespace CustomerPatronage.Prediction
{
    public class Predictor
    {
        private MLContext mlContext = new MLContext();
        private Dictionary<string, List<PurchasePredictionInput>> customerHistory = new();

        public Predictor()
        {
            LoadCustomerHistory();
        }

        private (ITransformer Model, ModelMetadata ModelMetadata) LoadModel(
            string modelName,
            int predictionMonthsWindow
        )
        {
            var filePaths = new FileManager().GetFilePaths(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );

            ITransformer? model = default;
            ModelMetadata? modelMetadata = default;

            Action<string, Action<string>> loadFile = (targetFilePath, fnLoadAction) =>
            {
                if (File.Exists(targetFilePath))
                {
                    fnLoadAction(targetFilePath);
                }
                else
                {
                    throw new FileNotFoundException("File not found", targetFilePath);
                }
            };

            loadFile(filePaths.ModelPath, (modelPath) =>
            {
                model = mlContext.Model.Load(modelPath, out var _);
            });

            loadFile(filePaths.MetadataPath, (metadataPath) =>
            {
                modelMetadata = JsonSerializer.Deserialize<ModelMetadata>(File.ReadAllText(metadataPath));
            });

            return (model!, modelMetadata!);
        }

        public IEnumerable<PurchasePredictionOutput> PredictMultipleCustomer(
            IEnumerable<string> customerIds,
            string modelName,
            int predictionMonthsWindow)
        {
            if (customerIds == null || !customerIds.Any())
            {
                throw new ArgumentException(paramName: $"{nameof(customerIds)}", message: $"{nameof(customerIds)} cannot be null or empty");
            }

            var modelAndMetadata = LoadModel(modelName: modelName, predictionMonthsWindow: predictionMonthsWindow);

            return customerIds.Select(customerId => PredictSingleCustomer(
                customerId: customerId,
                modelAndMetadata: modelAndMetadata
            ));
        }

        public PurchasePredictionOutput PredictSingleCustomer(
            string customerId,
            string modelName,
            int predictionMonthsWindow)
        {
            var modelAndMetadata = LoadModel(modelName: modelName, predictionMonthsWindow: predictionMonthsWindow);

            return PredictSingleCustomer(
                customerId: customerId,
                modelAndMetadata: modelAndMetadata
            );
        }

        private PurchasePredictionOutput PredictSingleCustomer(
            string customerId,
            (ITransformer Model, ModelMetadata ModelMetadata) modelAndMetadata)
        {
            if (!customerHistory.TryGetValue(customerId, out var history) || !history.Any())
            {
                // Return baseline prediction if no history is available
                return new PurchasePredictionOutput
                {
                    CustomerId = customerId,
                    ExpectedSpend = modelAndMetadata.ModelMetadata.AverageSpend,
                    PurchaseFrequency = modelAndMetadata.ModelMetadata.AverageFrequency,
                    IsBaseline = true // Indicate that this is a baseline prediction
                };
            }

            var model = modelAndMetadata.Model;
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PurchasePredictionInput, PurchasePredictionOutput>(model);

            var latestData = history.LastOrDefault(); // Use the most recent data

            if (latestData == null)
            {
                // If no data available, return baseline prediction
                return new PurchasePredictionOutput
                {
                    CustomerId = customerId,
                    ExpectedSpend = modelAndMetadata.ModelMetadata.AverageSpend,
                    PurchaseFrequency = modelAndMetadata.ModelMetadata.AverageFrequency,
                    IsBaseline = true // Indicate that this is a baseline prediction
                };
            }

            var prediction = predictionEngine.Predict(latestData);

            // If prediction output is below a certain threshold, return baseline prediction
            if (prediction.ExpectedSpend < 0.01f && prediction.PurchaseFrequency < 0.01f)
            {
                return new PurchasePredictionOutput
                {
                    CustomerId = customerId,
                    ExpectedSpend = modelAndMetadata.ModelMetadata.AverageSpend,
                    PurchaseFrequency = modelAndMetadata.ModelMetadata.AverageFrequency,
                    IsBaseline = true // Indicate that this is a baseline prediction
                };
            }

            // Return normal prediction
            prediction.CustomerId = customerId; // Set CustomerId for the output
            prediction.IsBaseline = false;
            return prediction;
        }

        private void LoadCustomerHistory()
        {
            var filePath = "customer_history.json";
            if (File.Exists(filePath))
            {
                customerHistory = JsonSerializer.Deserialize<Dictionary<string, List<PurchasePredictionInput>>>(File.ReadAllText(filePath)) ?? new();
            }
        }
    }
}

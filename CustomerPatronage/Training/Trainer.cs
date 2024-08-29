using System.Text.Json;
using CustomerPatronage.Persistance;
using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;
using System;
using System.IO;

namespace CustomerPatronage.Training
{
    public class Trainer
    {
        public event Action<float> OnTrainingProgress = (_) => { }; // Event to report training progress

        private readonly MLContext mlContext;

        public Trainer()
        {
            mlContext = new MLContext();
        }

        public void TrainModel(
            IEnumerable<HistoricalPurchaseData> data,
            string modelName,
            int predictionMonthsWindow, 
            Func<HistoricalPurchaseData, bool>? fnFilter = null)
        {
            if (data == null || !data.Any())
            {
                throw new ArgumentNullException(nameof(data), $"Model training cannot proceed if {nameof(data)} is null or empty");
            }

            data = fnFilter == null ? data : data.Where(fnFilter);

            // Calculate baseline averages for fallback
            var metadata = new ModelMetadata
            {
                AverageSpend = data.Average(d => d.Spend),
                AverageFrequency = data.GroupBy(d => d.CustomerId).Average(g => g.Count() / (float)predictionMonthsWindow),
            };

            var trainingData = PrepareTrainingData(data, predictionMonthsWindow);

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(PurchasePredictionInput.DaysSinceLastPurchase),
                nameof(PurchasePredictionInput.DaysBetweenPurchases),
                nameof(PurchasePredictionInput.CumulativeSpend),
                nameof(PurchasePredictionInput.PurchaseFrequency))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            var trainer = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var model = trainingPipeline.Fit(trainingData);

            SaveModelAndMetadata(
                historicalPurchaseDatas: data,
                inputSchema: trainingData.Schema,
                model: model,
                modelMetadata: metadata,
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );
        }

        private void SaveModelAndMetadata(
            IEnumerable<HistoricalPurchaseData> historicalPurchaseDatas,
            DataViewSchema inputSchema,
            ITransformer model,
            ModelMetadata modelMetadata,
            string modelName,
            int predictionMonthsWindow)
        {
            var fileManager = new FileManager();
            var trainingDataDirectory = fileManager.GetTrainingDataPath(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );

            try
            {
                if (Directory.Exists(trainingDataDirectory))
                {
                    Directory.Delete(trainingDataDirectory, recursive: true);
                }
                Directory.CreateDirectory(trainingDataDirectory);
            }
            catch (Exception ex)
            {
                // Log the error and handle it appropriately
                throw new InvalidOperationException("Failed to create or delete training data directory", ex);
            }

            var filepaths = fileManager.GetFilePaths(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );

            mlContext.Model.Save(model, inputSchema, filepaths.ModelPath);

            File.WriteAllText(filepaths.MetadataPath, JsonSerializer.Serialize(modelMetadata, new JsonSerializerOptions { WriteIndented = true }));

            SaveCustomerHistory(historicalPurchaseDatas, predictionMonthsWindow, filepaths.CustomerHistoryPath);
        }

        private IDataView PrepareTrainingData(
            IEnumerable<HistoricalPurchaseData> data,
            int predictionMonthsWindow)
        {
            var featureData = new List<PurchasePredictionInput>();
            var groupedData = data.GroupBy(d => d.CustomerId);

            foreach (var group in groupedData)
            {
                var sortedPurchases = group.OrderBy(d => d.PurchaseDate).ToList();
                for (int i = 0; i < sortedPurchases.Count - 1; i++)
                {
                    var currentPurchase = sortedPurchases[i];
                    var futurePurchases = sortedPurchases.Skip(i + 1)
                        .Where(p => p.PurchaseDate <= currentPurchase.PurchaseDate.AddMonths(predictionMonthsWindow))
                        .ToList();

                    if (!futurePurchases.Any()) continue;

                    var expectedSpend = futurePurchases.Sum(p => p.Spend);
                    var purchaseFrequency = futurePurchases.Count / (float)predictionMonthsWindow;

                    var predictionInput = new PurchasePredictionInput
                    {
                        CustomerId = currentPurchase.CustomerId,
                        CustomerName = currentPurchase.CustomerName,
                        DaysSinceLastPurchase = i == 0 ? 0 : (float)(currentPurchase.PurchaseDate - sortedPurchases[i - 1].PurchaseDate).TotalDays,
                        DaysBetweenPurchases = i == 0 ? 0 : (float)(currentPurchase.PurchaseDate - sortedPurchases[i - 1].PurchaseDate).TotalDays,
                        CumulativeSpend = sortedPurchases.Take(i + 1).Sum(p => p.Spend),
                        PurchaseFrequency = purchaseFrequency,
                        Label = expectedSpend
                    };

                    featureData.Add(predictionInput);
                }
            }

            return mlContext.Data.LoadFromEnumerable(featureData);
        }

        private void SaveCustomerHistory(
            IEnumerable<HistoricalPurchaseData> data, 
            int predictionMonthsWindow,
            string filePath)
        {
            var customerHistory = PrepareCustomerHistory(data, predictionMonthsWindow);
            File.WriteAllText(filePath, JsonSerializer.Serialize(customerHistory, new JsonSerializerOptions { WriteIndented = true }));
        }

        private Dictionary<string, List<PurchasePredictionInput>> PrepareCustomerHistory(
            IEnumerable<HistoricalPurchaseData> data,
            int predictionMonthsWindow)
        {
            var customerHistory = new Dictionary<string, List<PurchasePredictionInput>>();
            var groupedData = data.GroupBy(d => d.CustomerId);

            foreach (var group in groupedData)
            {
                var sortedPurchases = group.OrderBy(d => d.PurchaseDate).ToList();
                var history = new List<PurchasePredictionInput>();

                for (int i = 0; i < sortedPurchases.Count - 1; i++)
                {
                    var currentPurchase = sortedPurchases[i];
                    var futurePurchases = sortedPurchases.Skip(i + 1)
                        .Where(p => p.PurchaseDate <= currentPurchase.PurchaseDate.AddMonths(predictionMonthsWindow))
                        .ToList();

                    if (!futurePurchases.Any()) continue;

                    var expectedSpend = futurePurchases.Sum(p => p.Spend);
                    var purchaseFrequency = futurePurchases.Count / (float)predictionMonthsWindow;

                    history.Add(new PurchasePredictionInput
                    {
                        CustomerId = currentPurchase.CustomerId,
                        CustomerName = currentPurchase.CustomerName,
                        DaysSinceLastPurchase = i == 0 ? 0 : (float)(currentPurchase.PurchaseDate - sortedPurchases[i - 1].PurchaseDate).TotalDays,
                        DaysBetweenPurchases = i == 0 ? 0 : (float)(currentPurchase.PurchaseDate - sortedPurchases[i - 1].PurchaseDate).TotalDays,
                        CumulativeSpend = sortedPurchases.Take(i + 1).Sum(p => p.Spend),
                        PurchaseFrequency = purchaseFrequency,
                        Label = expectedSpend
                    });
                }

                customerHistory[group.Key] = history;
            }

            return customerHistory;
        }
    }
}

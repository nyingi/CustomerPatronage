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

        private readonly MLContext _mlContext;
        private readonly FileManager _fileManager;

        public Trainer()
        {
            _mlContext = new MLContext();
            _fileManager = new FileManager();
        }

        public void TrainModel(
            IEnumerable<HistoricalPurchaseData> data,
            string modelName,
            int predictionMonthsWindow, 
            Func<HistoricalPurchaseData, bool>? fnFilter = null)
        {
            ValidateData(data);

            var filteredData = fnFilter == null ? data : data.Where(fnFilter);

            var metadata = CalculateModelMetadata(filteredData, predictionMonthsWindow);

            var trainingData = PrepareTrainingData(filteredData, predictionMonthsWindow);

            ITransformer model;

            try
            {
                model = TrainModelPipeline(trainingData);
            }
            catch (Exception ex)
            {
                // Log the error and handle it appropriately
                throw new InvalidOperationException("Failed to train the model", ex);
            }

            try
            {
                SaveModelAndMetadata(
                    data,
                    trainingData.Schema,
                    model,
                    metadata,
                    modelName,
                    predictionMonthsWindow
                );
            }
            catch (Exception ex)
            {
                // Log the error and handle it appropriately
                throw new InvalidOperationException("Failed to save model and metadata", ex);
            }
        }

        private void ValidateData(IEnumerable<HistoricalPurchaseData> data)
        {
            if (data == null || !data.Any())
            {
                throw new ArgumentNullException(nameof(data), $"Model training cannot proceed if {nameof(data)} is null or empty");
            }
        }

        private ModelMetadata CalculateModelMetadata(IEnumerable<HistoricalPurchaseData> data, int predictionMonthsWindow)
        {
            return new ModelMetadata
            {
                AverageSpend = data.Average(d => d.Spend),
                AverageFrequency = data.GroupBy(d => d.CustomerId).Average(g => g.Count() / (float)predictionMonthsWindow),
            };
        }

        private ITransformer TrainModelPipeline(IDataView trainingData)
        {
            var dataProcessPipeline = _mlContext.Transforms.Concatenate("Features",
                nameof(PurchasePredictionInput.DaysSinceLastPurchase),
                nameof(PurchasePredictionInput.DaysBetweenPurchases),
                nameof(PurchasePredictionInput.CumulativeSpend),
                nameof(PurchasePredictionInput.PurchaseFrequency))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"));

            var trainer = _mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline.Fit(trainingData);
        }

        private IDataView PrepareTrainingData(
            IEnumerable<HistoricalPurchaseData> data,
            int predictionMonthsWindow)
        {
            var featureData = data.GroupBy(a => a.CustomerId)
                .SelectMany(group =>
                    GeneratePredictionInputs(group.OrderBy(d => d.PurchaseDate).ToList(), predictionMonthsWindow)
                ).ToList();

            return _mlContext.Data.LoadFromEnumerable(featureData);
        }

        private IEnumerable<PurchasePredictionInput> GeneratePredictionInputs(
            List<HistoricalPurchaseData> sortedPurchases, 
            int predictionMonthsWindow)
        {
            var predictionInputs = new List<PurchasePredictionInput>();

            for (int i = 0; i < sortedPurchases.Count - 1; i++)
            {
                var currentPurchase = sortedPurchases[i];
                var futurePurchases = sortedPurchases.Skip(i + 1)
                    .Where(p => p.PurchaseDate <= currentPurchase.PurchaseDate.AddMonths(predictionMonthsWindow))
                    .ToList();

                if (!futurePurchases.Any()) continue;

                var expectedSpend = futurePurchases.Sum(p => p.Spend);
                var purchaseFrequency = futurePurchases.Count / (float)predictionMonthsWindow;

                // Compute the number of days since the last purchase
                var daysSinceLastPurchase = i == 0 ? 0 : (float)(currentPurchase.PurchaseDate - sortedPurchases[i - 1].PurchaseDate).TotalDays;
                // Compute the number of days between purchases
                var daysBetweenPurchases = i == 0 ? 0 : daysSinceLastPurchase;
                // Compute the cumulative spend up to the current purchase
                var cumulativeSpend = sortedPurchases.Take(i + 1).Sum(p => p.Spend);
                // Compute purchase frequency
                var purchaseFreq = i == 0 ? 0 : i / (float)(currentPurchase.PurchaseDate - sortedPurchases.First().PurchaseDate).TotalDays;

                predictionInputs.Add(new PurchasePredictionInput
                {
                    CustomerId = currentPurchase.CustomerId,
                    CustomerName = currentPurchase.CustomerName,
                    DaysSinceLastPurchase = daysSinceLastPurchase,
                    DaysBetweenPurchases = daysBetweenPurchases,
                    CumulativeSpend = cumulativeSpend,
                    PurchaseFrequency = purchaseFrequency,
                    Label = expectedSpend
                });
            }

            return predictionInputs;
        }

        private void SaveModelAndMetadata(
            IEnumerable<HistoricalPurchaseData> historicalPurchaseDatas,
            DataViewSchema inputSchema,
            ITransformer model,
            ModelMetadata modelMetadata,
            string modelName,
            int predictionMonthsWindow)
        {
            var trainingDataDirectory = _fileManager.GetTrainingDataPath(modelName, predictionMonthsWindow);

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

            var filepaths = _fileManager.GetFilePaths(modelName, predictionMonthsWindow);

            try
            {
                _mlContext.Model.Save(model, inputSchema, filepaths.ModelPath);
                File.WriteAllText(filepaths.MetadataPath, JsonSerializer.Serialize(modelMetadata, new JsonSerializerOptions { WriteIndented = true }));
                SaveCustomerHistory(historicalPurchaseDatas, predictionMonthsWindow, filepaths.CustomerHistoryPath);
            }
            catch (Exception ex)
            {
                // Log the error and handle it appropriately
                throw new InvalidOperationException("Failed to save model and metadata files", ex);
            }
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
            return data
                .GroupBy(d => d.CustomerId)
                .ToDictionary(
                    group => group.Key,
                    group => GeneratePredictionInputs(group.OrderBy(d => d.PurchaseDate).ToList(), predictionMonthsWindow).ToList()
                );
        }
    }
}

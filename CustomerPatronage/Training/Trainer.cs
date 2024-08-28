using System.Text.Json;
using CustomerPatronage.Persistance;
using Microsoft.ML;



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

        private void TrainModel(
            IEnumerable<HistoricalPurchaseData> data,
            string modelName,
            int predictionMonthsWindow, 
            CancellationToken cancellationToken)
        {

            // Calculate baseline averages for fallback
            var metadata = new ModelMetadata
            {
                AverageSpend = data.Average(d => d.Spend),
                AverageFrequency = data.GroupBy(d => d.CustomerId).Average(g => g.Count() / (float)predictionMonthsWindow),
            };

            var trainingData = PrepareTrainingData(data,predictionMonthsWindow);
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(PurchasePredictionInput.DaysSinceLastPurchase),
                nameof(PurchasePredictionInput.DaysBetweenPurchases),
                nameof(PurchasePredictionInput.CumulativeSpend),
                nameof(PurchasePredictionInput.PurchaseFrequency))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            var trainer = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainingOperation = Task.Run(() =>
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    Console.WriteLine("Training cancelled.");
                    return;
                }

                var model = trainingPipeline.Fit(trainingData);
                
                
            }, cancellationToken);

            trainingOperation.Wait(cancellationToken);
        }

        private void Save(
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
            if(!Directory.Exists(trainingDataDirectory))
            {
                Directory.CreateDirectory(trainingDataDirectory);
            }
            var filepaths = new FileManager().GetFilePaths(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );
            mlContext.Model.Save(
                    model:model,
                    inputSchema: null,
                    filePath: filepaths.ModelPath);
            File.WriteAllText(
                path: filepaths.MetadataPath,
                contents: JsonSerializer.Serialize(
                    value: modelMetadata,
                    options: new JsonSerializerOptions
                    {
                        WriteIndented = true,
                    }));
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

                    if (futurePurchases.Count == 0) continue;

                    var expectedSpend = futurePurchases.Sum(p => p.Spend);
                    var purchaseFrequency = futurePurchases.Count / (float)predictionMonthsWindow;

                    var daysSinceLastPurchase = i == 0 ? 0 : (float)(currentPurchase.PurchaseDate - sortedPurchases[i - 1].PurchaseDate).TotalDays;
                    var daysBetweenPurchases = i == 0 ? 0 : daysSinceLastPurchase;
                    var cumulativeSpend = sortedPurchases.Take(i + 1).Sum(p => p.Spend);
                    var purchaseFreq = i == 0 ? 0 : i / (float)(currentPurchase.PurchaseDate - sortedPurchases.First().PurchaseDate).TotalDays;

                    featureData.Add(new PurchasePredictionInput
                    {
                        CustomerId = currentPurchase.CustomerId,
                        CustomerName = currentPurchase.CustomerName,
                        DaysSinceLastPurchase = daysSinceLastPurchase,
                        DaysBetweenPurchases = daysBetweenPurchases,
                        CumulativeSpend = (float)cumulativeSpend,
                        PurchaseFrequency = purchaseFrequency
                    });
                }
            }

            return mlContext.Data.LoadFromEnumerable(featureData);
        }

        

        
    }
}
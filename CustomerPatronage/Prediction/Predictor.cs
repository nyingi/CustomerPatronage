using System.Text.Json;
using CustomerPatronage.Persistance;
using CustomerPatronage.Training;
using Microsoft.ML;

namespace CustomerPatronage.Prediction
{
    public class Predictor
    {

        private MLContext mlContext = new MLContext();

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
            IEnumerable<PurchasePredictionInput> inputs,
            string modelName,
            int predictionMonthsWindow)

        {
            if(inputs == null || inputs.Any() == false)
            {
                throw new ArgumentException(paramName: $"{nameof(inputs)}",message: $"{nameof(inputs)} cannot be null or empty");
            }
            var modelAndMetadata = LoadModel(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );
            foreach (var item in inputs)
            {
                var result = PredictSingleCustomer(
                    input: item,
                    modelAndMetadata: modelAndMetadata
                );
                yield return result;
            }
        }
        public PurchasePredictionOutput PredictSingleCustomer(
            PurchasePredictionInput input,
            string modelName,
            int predictionMonthsWindow)
        {
            var modelAndMetadata = LoadModel(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );

            return PredictSingleCustomer(
                input: input,
                modelAndMetadata: modelAndMetadata
            );
        }
        private PurchasePredictionOutput PredictSingleCustomer(
            PurchasePredictionInput input,
            (ITransformer Model, ModelMetadata ModelMetadata) modelAndMetadata)
        {

            

            // Baseline averages for fallback
            var averageSpend = modelAndMetadata.ModelMetadata.AverageSpend;
            var averageFrequency = modelAndMetadata.ModelMetadata.AverageFrequency;

            var model = modelAndMetadata.Model;
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
using CustomerPatronage.Workers;
using Microsoft.ML;

namespace CustomerPatronage.Training
{
    public class Trainer
    {
        public void Train(TrainingData trainingData)
        {
            var context = new MLContext();

            var aggregatedTrainingData = new Loader().LoadAndAggregateData(trainingData.Data, context);
            var trainingDataView = context.Data.LoadFromEnumerable(aggregatedTrainingData);

            // Define and fit the model pipeline
            var pipeline = context.Transforms.Concatenate("Features", "DaysSinceLastPurchase", "TotalSpend", "AverageDaysBetweenPurchases", "PurchaseFrequency")
                .Append(context.Regression.Trainers.Sdca(labelColumnName: "PurchaseFrequency", maximumNumberOfIterations: 100))
                .Append(context.Transforms.Concatenate("Features", "TotalSpend")
                    .Append(context.Regression.Trainers.Sdca(labelColumnName: "ExpectedSpend", maximumNumberOfIterations: 100)));

            var model = pipeline.Fit(trainingDataView);
            

            var modelPath = $"{trainingData.ModelName}.zip";
            context.Model.Save(model, trainingDataView.Schema, modelPath);

        }
    }
}
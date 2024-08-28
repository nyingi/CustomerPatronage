namespace CustomerPatronage.Persistance
{
    internal class FileManager
    {
        private readonly string baseDirectory;

        public FileManager()
        {
            baseDirectory = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "customer-patronage"
            );
        }

        public (string ModelPath, string MetadataPath) GetFilePaths(string modelName,int predictionMonthsWindow)
        {
            var modelPath = GetModelPath(modelName: modelName, predictionMonthsWindow: predictionMonthsWindow);
            var metadataPath = GetMetadataPath(modelName: modelName, predictionMonthsWindow: predictionMonthsWindow);
            return (ModelPath: modelPath, MetadataPath: metadataPath);
        }

         public string GetTrainingDataPath(
            string modelName,
            int predictionMonthsWindow)
        {
            var destinationDirectory = Path.Combine(baseDirectory, $"{modelName}-months-window-{predictionMonthsWindow}");
            return destinationDirectory;
        }

        private string GetModelPath(string modelName,int predictionMonthsWindow)
        {
            return GetFinalPath(
                filename: "model.zip",
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow);
        }

        private string GetMetadataPath(string modelName,int predictionMonthsWindow)
        {
            return GetFinalPath(
                filename: "metadata.json",
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );
        }

        private string GetFinalPath(
            string filename,
            string modelName,
            int predictionMonthsWindow)
        {
            var destinationDirectory = GetTrainingDataPath(
                modelName: modelName,
                predictionMonthsWindow: predictionMonthsWindow
            );
            return Path.Combine(destinationDirectory, filename);
        }
    }
}
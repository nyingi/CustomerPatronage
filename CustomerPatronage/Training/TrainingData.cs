namespace CustomerPatronage.Training
{
    public class TrainingData
    {
        public string ModelName { get; set; } = string.Empty;

        public required IEnumerable<HistoricalPurchaseData> Data { get; set; } 
    }
}
namespace CustomerPatronage.Prediction
{
    public class PurchasePredictionOutput
    {
        public string CustomerId { get; set; } = string.Empty;
        public string CustomerName { get; set; } = string.Empty;
        public float ExpectedSpend { get; set; }
        public float PurchaseFrequency { get; set; }
        public bool IsBaseline { get; set; } // Flag to indicate if this is a baseline prediction
    }
}
namespace CustomerPatronage.Training
{
    public class PurchasePredictionInput
    {
        public string CustomerId { get; set; } = string.Empty;
        public string CustomerName { get; set; } = string.Empty;
        public float DaysSinceLastPurchase { get; set; }
        public float DaysBetweenPurchases { get; set; }
        public float CumulativeSpend { get; set; }
        public float PurchaseFrequency { get; set; }    
    }
}
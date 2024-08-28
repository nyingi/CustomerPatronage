namespace CustomerPatronage.DataContainers
{
    public class HistoricalPurchaseData
    {
        public string CustomerId { get; set; } = string.Empty;
        public string CustomerName { get; set; } = string.Empty;
        public DateTime PurchaseDate { get; set; }
        public decimal Spend { get; set; } 
    }
}
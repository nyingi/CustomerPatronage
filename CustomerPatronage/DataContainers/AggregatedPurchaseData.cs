using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CustomerPatronage.DataContainers
{
    public class AggregatedPurchaseData
    {
        public string CustomerId { get; set; } = string.Empty;
        public string CustomerName { get; set; } = string.Empty;
        public decimal AverageDaysBetweenPurchases { get; set; }
        public decimal TotalSpend { get; set; }
        public decimal PurchaseFrequency { get; set; }
        public decimal DaysSinceLastPurchase { get; set; } 
    }
}
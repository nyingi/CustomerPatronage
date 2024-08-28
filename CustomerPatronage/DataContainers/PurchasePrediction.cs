using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CustomerPatronage.DataContainers
{
    public class PurchasePrediction
    {
        public string CustomerId { get; set; } = string.Empty;
        public string CustomerName { get; set; } = string.Empty;
        public decimal ExpectedSpend { get; set; }
        public float PurchaseFrequency { get; set; }
    }
}
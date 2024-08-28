using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CustomerPatronage.DataContainers
{
    public class DiscrepancyResult
    {
        public string CustomerId { get; set; } = string.Empty;
        public string CustomerName { get; set; } = string.Empty;
        public decimal ActualSpend { get; set; }
        public decimal PredictedSpend { get; set; }
        public decimal DiscrepancySpend { get; set; }
        public float PredictedFrequency { get; set; }
        public float DiscrepancyFrequency { get; set; }
    }
}
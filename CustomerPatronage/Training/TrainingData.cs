using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CustomerPatronage.DataContainers;

namespace CustomerPatronage.Training
{
    public class TrainingData
    {
        public string ModelName { get; set; } = string.Empty;

        public required IEnumerable<HistoricalPurchaseData> Data { get; set; } 
    }
}
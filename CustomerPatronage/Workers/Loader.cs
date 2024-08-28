using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CustomerPatronage.DataContainers;
using Microsoft.ML;

namespace CustomerPatronage.Workers
{
    internal class Loader
    {
        public List<AggregatedPurchaseData> LoadAndAggregateData(IEnumerable<HistoricalPurchaseData> rawData, MLContext context)
        {
            // Aggregate data as described earlier
            return rawData.Select(data => new AggregatedPurchaseData
            {
                CustomerId = data.CustomerId,
                CustomerName = data.CustomerName,
                DaysSinceLastPurchase = (decimal)(DateTime.Now - data.PurchaseDate).TotalDays,
                TotalSpend = data.Spend,
                AverageDaysBetweenPurchases = CalculateAverageDaysBetweenPurchases(rawData, data.CustomerId),
                PurchaseFrequency = CalculatePurchaseFrequency(rawData, data.CustomerId)
            }).ToList();
        }

        decimal CalculateAverageDaysBetweenPurchases(IEnumerable<HistoricalPurchaseData> data, string customerId)
        {
            var purchases = data.Where(p => p.CustomerId == customerId).OrderBy(p => p.PurchaseDate).ToList();
            return purchases.Count > 1
                ? (decimal)purchases.Zip(purchases.Skip(1), (a, b) => (b.PurchaseDate - a.PurchaseDate).TotalDays).Average()
                : (decimal)(DateTime.Now - purchases.Last().PurchaseDate).TotalDays;
        }

        decimal CalculatePurchaseFrequency(IEnumerable<HistoricalPurchaseData> data, string customerId)
        {
            var purchases = data.Where(p => p.CustomerId == customerId).ToList();
            return purchases.Count / (decimal)(DateTime.Now - purchases.First().PurchaseDate).TotalDays;
        }
    }
}
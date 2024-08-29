using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CustomerPatronage.Training;

namespace CustomerPatronage.Tests.Training
{
    public class TrainerTests
    {
        [Fact]
        public void TrainingModelIsSuccessfullyBuilt()
        {
            var trainer = new Trainer();
            trainer.TrainModel(new List<HistoricalPurchaseData>
            {
                new HistoricalPurchaseData
                {
                    CustomerId = "1",
                    CustomerName = "Alice",
                    PurchaseDate = DateTime.Now.AddMonths(-2),
                    Spend = 9
                },new HistoricalPurchaseData
                {
                    CustomerId = "1",
                    CustomerName = "Alice",
                    PurchaseDate = DateTime.Now.AddMonths(-1),
                    Spend = 9
                },
                new HistoricalPurchaseData
                {
                    CustomerId = "1",
                    CustomerName = "Alice",
                    PurchaseDate = DateTime.Now.AddMonths(-3),
                    Spend = 9
                }
            }, "basic-test", 3);
        }
    }
}
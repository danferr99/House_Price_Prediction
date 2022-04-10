using Microsoft.ML.Data;

namespace HousePricePrediction
{
    public class HousePrice
    {
        [LoadColumn(0)]
        public long id;

        [LoadColumn(1)]
        public int price;

        [LoadColumn(2)]
        public int bedrooms;

        [LoadColumn(3)]
        public int bathrooms;

        [LoadColumn(4)]
        public float sqft_lot;

        [LoadColumn(5)]
        public int floors;
    }

    public class HousePriceFarePrediction
    {
        [ColumnName("Score")]
        public float price;
    }


}

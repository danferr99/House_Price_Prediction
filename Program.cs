using System;
using System.IO;
using Microsoft.ML;

namespace HousePricePrediction
{
    class Program
    {

        static readonly string _dataPath = Path.Combine("C:\\Users\\danie\\Dropbox\\FACULDADE\\INTELIGÊNCIA ARTIFICIAL\\Atividade IA - CASAS\\HousePricePrediction\\Data", "house.csv");
        static readonly string _modelPath = Path.Combine("C:\\Users\\danie\\Dropbox\\FACULDADE\\INTELIGÊNCIA ARTIFICIAL\\Atividade IA - CASAS\\HousePricePrediction\\Data", "house.zip");
        static readonly string _trainDataPath = Path.Combine("C:\\Users\\danie\\Dropbox\\FACULDADE\\INTELIGÊNCIA ARTIFICIAL\\Atividade IA - CASAS\\HousePricePrediction\\Data", "house.zip");

        static void Main(string[] args)
        {
            Console.WriteLine(Environment.CurrentDirectory);

            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string _dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<HousePrice>(_dataPath, hasHeader: true, separatorChar: ';');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "price")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "idEncoded", inputColumnName: "id"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "priceEncoded", inputColumnName: "price"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "bedroomsEncoded", inputColumnName: "bedrooms"))
                    .Append(mlContext.Transforms.Concatenate("bathroomsEncoded", "sqft_lotEncoded", "floorsEncoded"))
                    .Append(mlContext.Regression.Trainers.FastTree());

            Console.WriteLine("=============== Create and Train the Model ===============");

            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<HousePrice>(_trainDataPath, hasHeader: true, separatorChar: ';');

            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            
            var predictionFunction = mlContext.Model.CreatePredictionEngine<HousePrice,HousePriceFarePrediction>(model);
           
            var HousePriceSample = new HousePrice()
            {
                id = 7129300520,
                price = 221900,
                bedrooms = 3,
                bathrooms = 1,
                sqft_lot = 5650,
                floors = 1
            };
            var prediction = predictionFunction.Predict(HousePriceSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.price:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
package NeuralNetCPUv;

public class Main {

    public static void main(String[] args) {
        int numTrain = 120;
        int numTest = 30;

        TrainDataset train = new TrainDataset(numTrain, 4, 3);
        TestDataset test = new TestDataset(numTest, 4, 3);

        DataUtils.splitDataset("iris.csv", numTrain, numTest, train, test);

        normalize(train);
        normalize(test);

        System.out.println("Dataset loaded, shuffled and normalized.");
        System.out.printf("Train examples: %d, Test examples: %d%n", train.numExamples, test.numExamples);

        int inputSize = 4;
        int[] hiddenSizes = {10, 8};
        int outputSize = 3;
        float learningRate = 0.01f;

        NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate);
        nn.printStructure();

        int epochs = 300;
        nn.train(train, epochs);


        float testAcc = NeuralNetwork.evaluate(test, nn);
        System.out.printf("Test Accuracy: %.2f%%%n", testAcc * 100);

        float[] sample = train.features.data[0];
        float[] prediction = nn.forward(sample);
        System.out.print("Prediction for first training sample: ");
        for (float p : prediction) System.out.printf("%.3f ", p);
        System.out.println();
    }

    private static void normalize(TrainDataset dataset) {
        for (int i = 0; i < dataset.numExamples; i++) {
            for (int j = 0; j < dataset.features.data[0].length; j++) {
                dataset.features.data[i][j] /= 8.0f;
            }
        }
    }

    private static void normalize(TestDataset dataset) {
        for (int i = 0; i < dataset.numExamples; i++) {
            for (int j = 0; j < dataset.features.data[0].length; j++) {
                dataset.features.data[i][j] /= 8.0f;
            }
        }
    }
}

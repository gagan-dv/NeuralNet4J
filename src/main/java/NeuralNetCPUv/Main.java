package NeuralNetCPUv;

/**
 * Main entry point of the Neural Network program.
 *
 * This program:
 *  1. Loads the Iris dataset (from resources/iris.csv).
 *  2. Splits it into training and testing sets.
 *  3. Normalizes feature values.
 *  4. Builds a Neural Network with hidden layers.
 *  5. Trains the network for a number of epochs.
 *  6. Evaluates the trained model on test data.
 *  7. Runs a prediction on a sample input.
 */
public class Main {

    public static void main(String[] args) {
        // === Step 1: Define dataset sizes ===
        int numTrain = 120;   // number of training samples
        int numTest = 30;     // number of testing samples

        // Create dataset objects
        TrainDataset train = new TrainDataset(numTrain, 4, 3);
        TestDataset test = new TestDataset(numTest, 4, 3);

        // === Step 2: Load and split dataset ===
        DataUtils.splitDataset("iris.csv", numTrain, numTest, train, test);

        // === Step 3: Normalize features ===
        normalize(train);
        normalize(test);

        // Print confirmation
        System.out.println("Dataset loaded, shuffled and normalized.");
        System.out.printf("Train examples: %d, Test examples: %d%n",
                train.numExamples, test.numExamples);

        // === Step 4: Define Neural Network structure ===
        int inputSize = 4;                 // 4 features per flower
        int[] hiddenSizes = {10, 8};       // 2 hidden layers: 10 neurons and 8 neurons
        int outputSize = 3;                // 3 classes (setosa, versicolor, virginica)
        float learningRate = 0.01f;        // step size for weight updates

        // Create the Neural Network
        NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate);

        // Print model structure
        nn.printStructure();

        // === Step 5: Train the model ===
        int epochs = 300;  // number of training passes over the dataset
        nn.train(train, epochs);

        // === Step 6: Evaluate model on test data ===
        float testAcc = NeuralNetwork.evaluate(test, nn);
        System.out.printf("Test Accuracy: %.2f%%%n", testAcc * 100);

        // === Step 7: Run prediction on a sample input ===
        float[] sample = train.features.data[0];   // take first training sample
        float[] prediction = nn.forward(sample);   // forward pass to get probabilities

        System.out.print("Prediction for first training sample: ");
        for (int i = 0; i < prediction.length; i++) {
            System.out.printf("%.3f ", prediction[i]);
        }
        System.out.println();
    }

    /**
     * Normalizes features in the training dataset by dividing by 8.0.
     * Ensures values are in a smaller range for faster learning.
     *
     * @param dataset training dataset
     */
    private static void normalize(TrainDataset dataset) {
        for (int i = 0; i < dataset.numExamples; i++) {
            for (int j = 0; j < dataset.features.data[0].length; j++) {
                dataset.features.data[i][j] = dataset.features.data[i][j] / 8.0f;
            }
        }
    }

    /**
     * Normalizes features in the testing dataset by dividing by 8.0.
     *
     * @param dataset testing dataset
     */
    private static void normalize(TestDataset dataset) {
        for (int i = 0; i < dataset.numExamples; i++) {
            for (int j = 0; j < dataset.features.data[0].length; j++) {
                dataset.features.data[i][j] = dataset.features.data[i][j] / 8.0f;
            }
        }
    }
}

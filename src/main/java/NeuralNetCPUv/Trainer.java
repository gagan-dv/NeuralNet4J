package NeuralNetCPUv;

/**
 * Trainer class handles the training loop and evaluation of a NeuralNetwork.
 */
public class Trainer {

    private NeuralNetwork network;
    private float learningRate;

    /**
     * Creates a new Trainer for a given neural network.
     *
     * @param network      the neural network to train
     * @param learningRate the learning rate for weight updates
     */
    public Trainer(NeuralNetwork network, float learningRate) {
        this.network = network;
        this.learningRate = learningRate;
    }

    /**
     * Trains the neural network on the given dataset for a number of epochs.
     *
     * @param trainData the training dataset
     * @param epochs    number of passes over the dataset
     */
    public void train(TrainDataset trainData, int epochs) {
        for (int epoch = 1; epoch <= epochs; epoch++) {

            // Shuffle dataset at the start of each epoch
            DataUtils.shuffleDataset(trainData.features, trainData.labels);

            float totalLoss = 0f;

            // Go through each training sample
            for (int i = 0; i < trainData.numExamples; i++) {
                float[] input = trainData.features.data[i];
                float[] target = trainData.labels.data[i];

                // Forward pass
                float[] predicted = network.forward(input);

                // Accumulate loss
                totalLoss += NeuralNetwork.crossEntropyLoss(predicted, target);

                // Backpropagation
                network.trainSample(input, target);
            }

            // Report average loss after each epoch
            System.out.printf("Epoch %d: Loss = %.4f%n", epoch, totalLoss / trainData.numExamples);
        }
    }

    /**
     * Evaluates the network’s accuracy on the test dataset.
     *
     * @param testData the dataset used for evaluation
     * @return accuracy as a fraction between 0 and 1
     */
    public float evaluate(TestDataset testData) {
        int correct = 0;

        // Go through each test example
        for (int i = 0; i < testData.numExamples; i++) {
            float[] predicted = network.forward(testData.features.data[i]);
            int predLabel = NeuralNetwork.argMax(predicted);
            int trueLabel = NeuralNetwork.argMax(testData.labels.data[i]);

            // Explicit if condition
            if (predLabel == trueLabel) {
                correct++;
            } else {
                // No increment if incorrect, but explicit branch keeps logic clear
                correct += 0;
            }
        }

        return (float) correct / testData.numExamples;
    }
}



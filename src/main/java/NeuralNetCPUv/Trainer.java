package NeuralNetCPUv;

public class Trainer {

    private NeuralNetwork network;
    private float learningRate;

    public Trainer(NeuralNetwork network, float learningRate) {
        this.network = network;
        this.learningRate = learningRate;
    }

    public void train(TrainDataset trainData, int epochs) {
        for (int epoch = 1; epoch <= epochs; epoch++) {

            DataUtils.shuffleDataset(trainData.features, trainData.labels);

            float totalLoss = 0f;

            for (int i = 0; i < trainData.numExamples; i++) {
                float[] input = trainData.features.data[i];
                float[] target = trainData.labels.data[i];


                float[] predicted = network.forward(input);


                totalLoss += NeuralNetwork.crossEntropyLoss(predicted, target);

                network.trainSample(input, target);
            }

            System.out.printf("Epoch %d: Loss = %.4f%n", epoch, totalLoss / trainData.numExamples);
        }
    }


    public float evaluate(TestDataset testData) {
        int correct = 0;

        for (int i = 0; i < testData.numExamples; i++) {
            float[] predicted = network.forward(testData.features.data[i]);
            int predLabel = NeuralNetwork.argMax(predicted);
            int trueLabel = NeuralNetwork.argMax(testData.labels.data[i]);
            if (predLabel == trueLabel) correct++;
        }

        return (float) correct / testData.numExamples;
    }
}

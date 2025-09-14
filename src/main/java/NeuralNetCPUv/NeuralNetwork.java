package NeuralNetCPUv;

import java.util.Random;
import java.util.function.Function;

public class NeuralNetwork {

    public Layer[] layers;
    private float learningRate;

    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, float learningRate) {
        this.learningRate = learningRate;
        int totalLayers = hiddenSizes.length + 1; // hidden + output
        layers = new Layer[totalLayers];

        int prevSize = inputSize;
        for (int i = 0; i < hiddenSizes.length; i++) {
            // ReLU hidden layers
            layers[i] = new Layer(prevSize, hiddenSizes[i],
                    x -> Math.max(0f, x),
                    z -> z > 0f ? 1f : 0f);
            prevSize = hiddenSizes[i];
        }

        layers[totalLayers - 1] = new Layer(prevSize, outputSize,
                x -> x,
                null);
    }

    public float[] forward(float[] input) {
        float[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }

        return Activations.softmax(output);
    }

    public void train(TrainDataset dataset, int epochs) {
        for (int e = 0; e < epochs; e++) {
            RandomUtil.shuffle(dataset.features.data, dataset.labels.data);
            float totalLoss = 0f;

            for (int i = 0; i < dataset.numExamples; i++) {
                float[] input = dataset.features.data[i];
                float[] target = dataset.labels.data[i];

                float[] output = forward(input);
                totalLoss += crossEntropyLoss(output, target);
                trainSample(input, target);
            }

            if ((e + 1) % 10 == 0)
                System.out.printf("Epoch %d: Loss = %.4f%n", e + 1, totalLoss / dataset.numExamples);
        }
    }

    public void trainSample(float[] input, float[] target) {
        float[] softmaxOut = forward(input);
        float[] delta = new float[softmaxOut.length];
        for (int i = 0; i < delta.length; i++) delta[i] = softmaxOut[i] - target[i];


        for (int l = layers.length - 1; l >= 0; l--) {
            Layer layer = layers[l];
            float[] dAprev = layer.backwardFromDZ(delta, learningRate);

            if (l > 0) {
                Layer prevLayer = layers[l - 1];
                float[] dZprev = new float[dAprev.length];
                for (int i = 0; i < dZprev.length; i++) {
                    float deriv = prevLayer.activationDerivative != null
                            ? prevLayer.activationDerivative.apply(prevLayer.lastZ[i])
                            : 1f;
                    dZprev[i] = dAprev[i] * deriv;
                }
                delta = dZprev;
            }
        }
    }

    public void printStructure() {
        System.out.println("Neural Network Structure:");
        for (int i = 0; i < layers.length; i++) {
            Layer l = layers[i];
            System.out.printf("Layer %d: inputs=%d, outputs=%d%n", i + 1, l.inputSize, l.outputSize);
        }
    }

    public static float crossEntropyLoss(float[] predicted, float[] target) {
        float loss = 0f;
        for (int i = 0; i < predicted.length; i++)
            loss -= target[i] * Math.log(predicted[i] + 1e-10f);
        return loss;
    }

    public static float evaluate(TestDataset test, NeuralNetwork nn) {
        int correct = 0;
        for (int i = 0; i < test.numExamples; i++) {
            float[] pred = nn.forward(test.features.data[i]);
            if (argMax(pred) == argMax(test.labels.data[i]))
                correct++;
        }
        return (float) correct / test.numExamples;
    }

    public static int argMax(float[] arr) {
        int idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }
}



package NeuralNetCPUv;

import java.util.Random;
import java.util.function.Function;

public class Layer {
    public int inputSize;
    public int outputSize;
    public Matrix weights;

    public float[] biases;
    public float[] lastInput;
    public float[] lastZ;
    public float[] lastOutput;

    public Function<Float, Float> activation;
    public Function<Float, Float> activationDerivative;

    private Random rand = new Random();

    public Layer(int inputSize, int outputSize,
                 Function<Float, Float> activation,
                 Function<Float, Float> activationDerivative) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        this.activationDerivative = activationDerivative;
        this.weights = new Matrix(inputSize, outputSize);
        this.biases = new float[outputSize];
        this.weights.randomizeUniform();

        for (int i = 0; i < outputSize; i++)
            this.biases[i] = 0.0f;
    }
    public float[] forward(float[] input) {
        this.lastInput = input.clone(); // keep a copy
        float[] z = Matrix.multiplyVector(weights.data, input); // length = outputSize
        for (int i = 0; i < outputSize; i++) z[i] += biases[i];
        this.lastZ = z.clone();
        this.lastOutput = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            this.lastOutput[i] = activation.apply(z[i]);
        }
        return lastOutput;
    }

    public float[] backwardFromDZ(float[] dZ, float learningRate) {
        float[] dAprev = new float[inputSize];

        for (int i = 0; i < inputSize; i++) {
            float sum = 0f;
            for (int j = 0; j < outputSize; j++) {
                sum += weights.data[i][j] * dZ[j];
            }
            dAprev[i] = sum;
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                float grad = lastInput[i] * dZ[j];
                weights.data[i][j] -= learningRate * grad;
            }
        }

        for (int j = 0; j < outputSize; j++) {
            biases[j] -= learningRate * dZ[j];
        }

        return dAprev;
    }
}

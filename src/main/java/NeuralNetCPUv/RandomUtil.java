package NeuralNetCPUv;

import java.util.Random;

public class RandomUtil {
    private static final Random rand = new Random();

    // Optional: allow reproducibility when debugging
    public static void setSeed(long seed) {
        rand.setSeed(seed);
    }

    /**
     * Shuffles two arrays in parallel (keeps inputs and outputs aligned).
     * Uses Fisher–Yates shuffle algorithm.
     */
    public static void shuffle(float[][] inputs, float[][] outputs) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Inputs and outputs must have the same length.");
        }

        for (int i = inputs.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            // Swap inputs
            float[] tmpInput = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = tmpInput;

            // Swap outputs
            float[] tmpOutput = outputs[i];
            outputs[i] = outputs[j];
            outputs[j] = tmpOutput;
        }
    }
}



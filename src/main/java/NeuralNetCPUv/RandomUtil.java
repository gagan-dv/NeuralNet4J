package NeuralNetCPUv;

import java.util.Random;

public class RandomUtil {
    private static final Random rand = new Random();

    public static void shuffle(float[][] inputs, float[][] outputs) {
        for (int i = inputs.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            float[] tmpInput = inputs[i]; inputs[i] = inputs[j]; inputs[j] = tmpInput;
            float[] tmpOutput = outputs[i]; outputs[i] = outputs[j]; outputs[j] = tmpOutput;
        }
    }
}

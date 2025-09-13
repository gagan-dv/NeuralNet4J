package NeuralNetCPUv;

import java.util.Random;

public class Vector {

    private static Random rand = new Random();

    public static float[] alloc(int size, boolean zeroInit) {
        float[] vec = new float[size];
        if (zeroInit) {
            for (int i = 0; i < size; i++) vec[i] = 0.0f;
        }
        return vec;
    }

    public static float[] copy(float[] vec) {
        float[] newVec = new float[vec.length];
        System.arraycopy(vec, 0, newVec, 0, vec.length);
        return newVec;
    }


    public static float dot(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }


    public static float[] add(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] + b[i];
        return result;
    }

    public static float[] subtract(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] - b[i];
        return result;
    }


    public static float[] multiplyElem(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] * b[i];
        return result;
    }


    public static float[] divideElem(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] / b[i];
        return result;
    }


    public static float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] * scalar;
        return result;
    }


    public static float norm(float[] a) {
        float sum = 0.0f;
        for (float v : a) sum += v * v;
        return (float) Math.sqrt(sum);
    }

    public static float[] normalize(float[] a) {
        float mag = norm(a);
        if (mag == 0) return copy(a);
        return scale(a, 1.0f / mag);
    }

    public static void zeros(float[] vec) {
        for (int i = 0; i < vec.length; i++) vec[i] = 0.0f;
    }


    public static void ones(float[] vec) {
        for (int i = 0; i < vec.length; i++) vec[i] = 1.0f;
    }

    public static void randomUniform(float[] vec) {
        for (int i = 0; i < vec.length; i++)
            vec[i] = rand.nextFloat() * 2f - 1f;
    }

    public static void randomNormal(float[] vec) {
        for (int i = 0; i < vec.length; i++)
            vec[i] = (float) rand.nextGaussian();
    }

    public static void applyFunction(float[] vec, Function func) {
        for (int i = 0; i < vec.length; i++) vec[i] = func.apply(vec[i]);
    }

    public interface Function {
        float apply(float x);
    }

    public static void print(float[] vec) {
        System.out.print("[");
        for (int i = 0; i < vec.length; i++) {
            System.out.printf("%8.4f", vec[i]);
            if (i != vec.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }
}

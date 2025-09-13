package NeuralNetCPUv;

import java.util.Random;

public class Matrix {
    public int rows;
    public int cols;
    public float[][] data;

    private static Random rand = new Random();


    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];
    }


    public Matrix() {
        this.rows = 0;
        this.cols = 0;
        this.data = null;
    }


    public static float[][] multiply(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = B[0].length;
        int inner = B.length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < inner; k++)
                    result[i][j] += A[i][k] * B[k][j];
        return result;
    }


    public static float[] multiplyVector(float[][] mat, float[] vec) {
        int rows = mat.length;
        int cols = mat[0].length;
        float[] out = new float[cols];
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                out[j] += mat[i][j] * vec[i];
        return out;
    }


    public static float[][] transpose(float[][] mat) {
        int rows = mat.length;
        int cols = mat[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                transposed[j][i] = mat[i][j];
        return transposed;
    }


    public static float[][] add(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out[i][j] = A[i][j] + B[i][j];
        return out;
    }


    public static float[][] subtract(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out[i][j] = A[i][j] - B[i][j];
        return out;
    }


    public static float[][] multiplyElem(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out[i][j] = A[i][j] * B[i][j];
        return out;
    }


    public static float[][] multiplyScalar(float[][] A, float scalar) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out[i][j] = A[i][j] * scalar;
        return out;
    }


    public void zeros() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = 0.0f;
    }


    public void ones() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = 1.0f;
    }


    public void randomizeUniform() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = rand.nextFloat() * 2f - 1f;
    }


    public void randomizeNormal() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = (float) rand.nextGaussian();
    }


    public static float[][] copy(float[][] A) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(A[i], 0, out[i], 0, cols);
        return out;
    }


    public void applyFunction(Function func) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = func.apply(data[i][j]);
    }


    public interface Function {
        float apply(float x);
    }

    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                System.out.printf("%8.4f ", data[i][j]);
            System.out.println();
        }
    }

    public float[] multiplyVec(float[] vec) {
        float[] out = new float[cols];
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                out[j] += data[i][j] * vec[i];
        return out;
    }
}

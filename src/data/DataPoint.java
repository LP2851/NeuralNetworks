package data;

import static mnist.MnistReader.MNIST_IMAGE_TOTAL_SIZE;
import static mnist.MnistReader.MNIST_IMAGE_WIDTH_HEIGHT;

public class DataPoint {

    public double[] inputs, expectedOutputs;
    public int label;

    public DataPoint(double[] inputs, double[] expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }
    public DataPoint(double[] inputs, double[] expectedOutputs, int label) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        this.label = label;
    }

    public static DataPoint createMnistDataPoint(double[][] inputImage, int expectedOutput) {
        double[] inputs = new double[MNIST_IMAGE_TOTAL_SIZE];

        for (int x = 0; x < MNIST_IMAGE_WIDTH_HEIGHT; x++) {
            for (int y = 0; y < MNIST_IMAGE_WIDTH_HEIGHT; y++) {
                inputs[x * MNIST_IMAGE_WIDTH_HEIGHT + y] = inputImage[x][y] / 255d;
            }
        }

        double[] out = new double[10];
        out[expectedOutput] = 1d;

        return new DataPoint(inputs, out, expectedOutput);
    }
}

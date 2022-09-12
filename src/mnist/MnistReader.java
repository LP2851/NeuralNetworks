package mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MnistReader {

    public static final int MNIST_CLASSIFICATIONS_TOTAL = 10;
    public static final int MNIST_IMAGE_WIDTH_HEIGHT = 28;
    public static final int MNIST_IMAGE_TOTAL_SIZE = MNIST_IMAGE_WIDTH_HEIGHT * MNIST_IMAGE_WIDTH_HEIGHT;

    public static final String TESTING_DATA_PATH = "resources\\mnist\\t10k-images.idx3-ubyte";
    public static final String TESTING_LABELS_PATH = "resources\\mnist\\t10k-labels.idx1-ubyte";
    public static final String TRAINING_DATA_PATH = "resources\\mnist\\train-images.idx3-ubyte";
    public static final String TRAINING_LABELS_PATH = "resources\\mnist\\train-labels.idx1-ubyte";


    public static MnistDataPoint[] getTrainingDataPoints()
            throws IOException {
        return readData(TRAINING_DATA_PATH, TRAINING_LABELS_PATH);
    }

    public static MnistDataPoint[] getTestingDataPoints()
            throws IOException {
        return readData(TESTING_DATA_PATH, TESTING_LABELS_PATH);
    }

    private static MnistDataPoint[] readData(String dataFilePath, String labelFilePath)
            throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));

        int magicNumber = dataInputStream.readInt();
        int numOfItems = dataInputStream.readInt();
        int rows = dataInputStream.readInt();
        int cols = dataInputStream.readInt();

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));

        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        var data = new MnistDataPoint[numOfItems];

        for(int i = 0; i < numOfItems; i++) {
            double[][] image = new double[MNIST_IMAGE_WIDTH_HEIGHT][MNIST_IMAGE_WIDTH_HEIGHT];
            int label = labelInputStream.readUnsignedByte();

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    image[r][c]= dataInputStream.readUnsignedByte();
                }
            }
            data[i] = MnistDataPoint.createMnistDataPoint(image, label);
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
}

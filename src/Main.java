import data.DataPoint;
import mnist.MnistPanel;
import mnist.MnistReader;

import java.io.IOException;

/**
 *
 * @version 0.1
 * @author lucas
 */
public class Main {

    private static DataPoint[] trainingData, testingData;

    /**
     * Runs the neural network
     * @param args
     */
    public static void main(String[] args) {
        // Loading the datasets from the files.
        loadDataSets();

        // Displaying the images loaded (for debugging purposes)
        try {
            MnistPanel.showImages(trainingData);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // network.NeuralNetwork nn = new network.NeuralNetwork(28 * 28, 10, 10);
    }

    /**
     * Loads the dataset from the files.
     */
    private static void loadDataSets() {
        try {
            trainingData = MnistReader.getTrainingDataPoints();
            testingData = MnistReader.getTestingDataPoints();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
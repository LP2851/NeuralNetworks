import data.DataPoint;
import mnist.MnistPanel;
import mnist.MnistReader;

import java.io.FileNotFoundException;
import java.io.IOException;

public class Main {

    private static DataPoint[] trainingData, testingData;

    public static void main(String[] args) {
        // System.out.println("Hello world!");
        loadDataSets();
        try {
            MnistPanel.showImages(trainingData);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        // network.NeuralNetwork nn = new network.NeuralNetwork(28 * 28, 10, 10);
    }

    private static void loadDataSets() {
        try {
            trainingData = MnistReader.getTrainingDataPoints();
            testingData = MnistReader.getTestingDataPoints();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
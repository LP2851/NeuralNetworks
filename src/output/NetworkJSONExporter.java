package output;

import network.Layer;
import network.NeuralNetwork;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.FileWriter;
import java.io.IOException;

/**
 * For exporting neural networks once complete
 *
 * @author lucas
 * @version 0.1
 */
public class NetworkJSONExporter {

    private static final String FILE_TYPE = ".neuralnetwork";
    private static final String PATH = "E:/";

    /**
     * Generates a JSON file storing the passed neural network
     *
     * @param network     The network to be saved
     * @param networkName The network's name (the filename)
     * @return If the save was successful
     */
    public static boolean saveNetwork(NeuralNetwork network, String networkName) {
        var inputLayer = network.layers[0];
        var outputLayers = network.layers[network.layers.length - 1];

        JSONObject obj = new JSONObject();

        // General info
        obj.put("name", networkName);
        obj.put("inputCount", inputLayer.getNumNodesIn());
        obj.put("outputCount", outputLayers.getNumNodesOut());

        // Adding all the layers (in order)
        var allLayers = new JSONArray();
        for (var layer : network.layers)
            allLayers.add(generateLayerObject(layer));
        obj.put("layers", allLayers);

        // Writing to the file
        try {
            FileWriter f = new FileWriter(PATH + networkName + FILE_TYPE);
            f.write(obj.toJSONString());
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Takes a layer and generates an object storing all the relevant information
     *
     * @param layer Layer to be converted into JSONObject
     * @return JSONObject representing the weights and biases for the layer
     */
    private static JSONObject generateLayerObject(Layer layer) {
        var obj = new JSONObject();

        var weights = new JSONArray();

        for (double[] node : layer.weights) {
            var w = new JSONArray();
            for (double n : node) {
                w.add(n);
            }
            weights.add(w);
        }
        obj.put("weights", weights);

        var biases = new JSONArray();
        for (var b : layer.biases)
            biases.add(b);
        obj.put("biases", biases);

        return obj;
    }

}

package data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @param <E> Extends DataPoint
 */
public class BatchHandler<E extends DataPoint> {

    private final List<E> dataset;
    int currentIndex = 0;
    private int batchNumber = 0;
    public int batchSize = 100;
    private int shuffleIndex = 0;
    private int maxShuffles = -1;

    public BatchHandler(int batchSize, E[] dataset, int maxShuffles, boolean startWithShuffle) {
        this.batchSize = batchSize;
        this.dataset = convertDatasetArrayToArrayList(dataset);
        this.maxShuffles = maxShuffles;
        if (startWithShuffle) {
            shuffle();
            shuffleIndex--;
        }

    }

    public BatchHandler(int batchSize, E[] dataset) {
        this.batchSize = batchSize;
        this.dataset = convertDatasetArrayToArrayList(dataset);
    }

    public E[] nextBatch() {
        if (batchNumber >= calculateTotalBatchNumber())
            return (E[]) new DataPoint[0];

        batchNumber++;
        int actualBatchSize = Math.min(dataset.size() - currentIndex, batchSize);

        DataPoint[] batch = new DataPoint[actualBatchSize];
        for (int i = 0; i < actualBatchSize; i++) {
            if (currentIndex > dataset.size() - 1) {
                if (shuffleIndex < maxShuffles)
                    shuffle();
                else
                    break;
            }

            batch[i] = dataset.get(currentIndex);
            currentIndex++;
        }

        if (currentIndex >= dataset.size() - 1 && shuffleIndex < maxShuffles)
            shuffle();

        return (E[]) batch;


    }

    private ArrayList<E> convertDatasetArrayToArrayList(E[] dataset) {
        return new ArrayList<>(Arrays.stream(dataset).toList());
    }

    private void shuffle() {
        Collections.shuffle(dataset);
        currentIndex = 0;
        shuffleIndex++;
    }

    public int getBatchNumber() {
        return batchNumber;
    }

    public int calculateTotalBatchNumber() {
        int size = (dataset.size() / batchSize) * maxShuffles;
        return size + ((dataset.size() % batchSize > 0) ? 1 : 0);
    }
}

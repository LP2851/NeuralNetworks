package data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class BatchHandler<E extends DataPoint> {

    private final List<E> dataset;
    public int batchSize = 1000;
    public boolean useShuffle = false;
    int currentIndex = 0;
    private int batchNumber = 0;
    private int maxShuffles = 0;
    private int shuffleIndex = 0;


    public BatchHandler(int batchSize, E[] dataset, boolean useShuffle, int maxShuffles) {
        this.batchSize = batchSize;
        this.dataset = new ArrayList<>(Arrays.stream(dataset).toList());
        this.useShuffle = useShuffle;
        this.maxShuffles = maxShuffles;
        if (useShuffle)
            shuffle();
    }

    public E[] nextBatch() {
        return nextBatch(batchSize);
    }

    public E[] nextBatch(int batchSize) {
        if (batchNumber >= calculateTotalBatchNumber())
            return (E[]) new DataPoint[0];
        batchNumber++;
        int actualBatchSize = Math.min(dataset.size() - currentIndex, batchSize);

        DataPoint[] batch = new DataPoint[actualBatchSize];
        for (int i = 0; i < actualBatchSize; i++) {

            if(currentIndex > dataset.size()-1) {
                if (shuffleIndex < maxShuffles)
                    shuffle();
                else
                    break;
            }

            batch[i] = dataset.get(currentIndex);
            currentIndex++;
        }

        if (currentIndex >= dataset.size()-1 && shuffleIndex < maxShuffles)
            shuffle();

        return (E[]) batch;
    }

    public void shuffle() {
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

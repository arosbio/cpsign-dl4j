package com.arosbio.ml.dl4j.eval;

import java.io.Closeable;
import java.io.IOException;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arosbio.commons.Stopwatch;

public class EarlyStopScoreListener implements EarlyStoppingListener<MultiLayerNetwork>, Closeable {
	private static final Logger LOGGER = LoggerFactory.getLogger(EarlyStopScoreListener.class);
	private static final String EPOCH_HEADER = "Epoch";
	private static final String TRAIN_SCORE_HEADER = "Train score";
	private static final String TEST_SCORE_HEADER = "Test score";
	private static final String RUNTIME_MS_HEADER = "Runtime (ms)";
	private static final char MISSING_VALUE = '-';


	private boolean scoreBasedOnTest;
	private CSVPrinter printer;
	private int epochInterval = 1;
	private DataSetLossCalculator trainScorer, testScorer;
	private Stopwatch watch = new Stopwatch();

	public EarlyStopScoreListener(Appendable out, boolean scoresBasedOnTest) throws IOException {
		printer = CSVFormat.DEFAULT.builder().setCommentMarker('#').setAutoFlush(true).build().print(out);
		this.scoreBasedOnTest = scoresBasedOnTest;
	}

	public EarlyStopScoreListener(Appendable out, boolean scoresBasedOnTest, Map<String,Object> params) throws IOException {
		this(out,scoresBasedOnTest);
		// Write the parameters first
		printer.printComment(toNiceString(params));
	}
	
	public EarlyStopScoreListener interval(int interval) {
		epochInterval = interval;
		return this;
	}

	public EarlyStopScoreListener testScorer(DataSetLossCalculator scorer) {
		this.testScorer = scorer;
		return this;
	}

	public EarlyStopScoreListener trainScorer(DataSetLossCalculator scorer) {
		this.trainScorer = scorer;
		return this;
	}
	
	public static String toNiceString(Map<?,?> map) {
		String withBrackets = map.toString();
		return withBrackets.substring(1, withBrackets.length()-1);
	}


	/**
	 * Writes the header 
	 * @param esConfig ignored
	 * @param net ignored
	 */
	@Override
	public void onStart(EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net) {
		try {
			printer.printRecord(EPOCH_HEADER,TRAIN_SCORE_HEADER,TEST_SCORE_HEADER,RUNTIME_MS_HEADER);
		} catch (IOException e) {
			LOGGER.debug("Failed writing header",e);
		}
		watch.start();
	}

	@Override
	public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<MultiLayerNetwork> esConfig,
			MultiLayerNetwork net) {
		// Check if evaluation should be done
		if (epochNum % epochInterval != 0)
			return;

		try {
			watch.stop();
			
			printer.printRecord(
					epochNum, // epoch
					getTrainScore(score, net), // Train
					getTestScore(score, net), // Test
					watch.elapsedTimeMillis() // Runtime
					);
		} catch (Exception e) {
			LOGGER.debug("Failed writing epoch score info",e);
		}
		watch.start();

	}

	private Object getTestScore(double givenScore, Model m) {
		// If we're using test-records when fitting the network
		if (scoreBasedOnTest)
			return givenScore;
		// if we have a test-scorer
		if (testScorer !=null)
			return testScorer.calculateScore(m);
		return MISSING_VALUE;
	}
	private Object getTrainScore(double givenScore, Model m) {
		// If we're using train-records when fitting the network
		if (! scoreBasedOnTest)
			return givenScore;
		// if we have a test-scorer
		if (trainScorer !=null)
			return trainScorer.calculateScore(m);
		return MISSING_VALUE;
	}

	@Override
	public void onCompletion(EarlyStoppingResult<MultiLayerNetwork> esResult) {
		close();
	}
	
	public void close() {
		try {
			printer.println();
			printer.close(true);
		} catch (IOException e) {
			LOGGER.debug("Failed closing the appendable where scores are printed",e);
		}
	}

}

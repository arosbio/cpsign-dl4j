package com.arosbio.ml.dl4j.eval;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

public class EpochScoreCollector extends BaseTrainingListener {
	
	// When to print
	private int epochInterval = 1;
	
	// Epoch state
	private int ep = 0;
	// Scores 
	private Map<Integer,Double> epochScores = new HashMap<>();
	
	/**
	 * Save the score for every epoch
	 */
	public EpochScoreCollector() {
	}
	
	/**
	 * Save the epoch score for every {@code epoch} 
	 * @param epoch the print interval, e.g. for each 10th epoch
	 */
	public EpochScoreCollector(int epoch) {
		this.epochInterval = epoch;
	}
	
	
	public void onEpochEnd(Model m) {
		if (ep % epochInterval == 0) {
			epochScores.put(ep, m.score());
		}
		ep++;
	}
	
	public Map<Integer,Double> getScores(){
		return epochScores;
	}

}

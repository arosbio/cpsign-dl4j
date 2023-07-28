/*
 * Copyright (C) Aros Bio AB.
 *
 * CPSign is an Open Source Software that is dual licensed to allow you to choose a license that best suits your requirements:
 *
 * 1) GPLv3 (GNU General Public License Version 3) with Additional Terms, including an attribution clause as well as a limitation to use the software for commercial purposes.
 *
 * 2) CPSign Proprietary License that allows you to use CPSign for commercial activities, such as in a revenue-generating operation or environment, or integrate CPSign in your proprietary software without worrying about disclosing the source code of your proprietary software, which is required if you choose to use the software under GPLv3 license. See arosbio.com/cpsign/commercial-license for details.
 */
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

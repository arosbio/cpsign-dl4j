package com.arosbio.ml.dl4j;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

public class SysOutEpochScoreListener extends BaseTrainingListener {
	
	// When to print
	private int interval = 1;
	
	// Epoch state
	int ep = 0;
	
	public SysOutEpochScoreListener() {
	}
	
	public SysOutEpochScoreListener(int printInterval) {
		this.interval = printInterval;
	}
	
	
	
	public void onEpochEnd(Model m) {
		if (ep % interval == 0) {
			System.out.println("Score epoch {"+ep+"} : {"+m.score()+"}");
		}
		
		ep++;
	}

}

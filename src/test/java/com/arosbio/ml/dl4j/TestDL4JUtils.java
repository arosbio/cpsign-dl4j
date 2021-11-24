package com.arosbio.ml.dl4j;

import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.learning.config.AdaBelief;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;

public class TestDL4JUtils {

	@Test
	public void testUpdaterConversion() {
		// SGD
		IUpdater updater = DL4JUtils.toUpdater("Sgd;1");
		Assert.assertTrue(updater instanceof Sgd);
		Assert.assertEquals(1., ((Sgd)updater).getLearningRate(),0.0001);

		// Ada belief
		updater = DL4JUtils.toUpdater("adaBelief;.5");
		Assert.assertTrue(updater instanceof AdaBelief);
		Assert.assertEquals(.5, ((AdaBelief)updater).getLearningRate(),0.0001);

		// Ada delta
		updater = DL4JUtils.toUpdater("adaDELTA;.25;4");
		Assert.assertTrue(updater instanceof AdaDelta);
		Assert.assertEquals(.25, ((AdaDelta)updater).getRho(),0.0001);
		Assert.assertEquals(4, ((AdaDelta)updater).getEpsilon(),0.0001);

		// Ada grad
		updater = DL4JUtils.toUpdater("adagrad;.5");
		Assert.assertTrue(updater instanceof AdaGrad);
		Assert.assertEquals(.5, ((AdaGrad)updater).getLearningRate(),0.0001);

		// Adam
		updater = DL4JUtils.toUpdater("adam;.15");
		Assert.assertTrue(updater instanceof Adam);
		Assert.assertEquals(.15, ((Adam)updater).getLearningRate(),0.0001);

		// Ada max
		updater = DL4JUtils.toUpdater("ADAMAX;;0.5;.1;na");
		Assert.assertTrue(updater instanceof AdaMax);
		Assert.assertEquals(AdaMax.DEFAULT_ADAMAX_LEARNING_RATE, ((AdaMax)updater).getLearningRate(),0.0001); // Default LR
		Assert.assertEquals(.5, ((AdaMax)updater).getBeta1(),0.0001); //Custom beta1
		Assert.assertEquals(.1, ((AdaMax)updater).getBeta2(),0.0001); //Custom beta2
		Assert.assertEquals(AdaMax.DEFAULT_ADAMAX_EPSILON, ((AdaMax)updater).getEpsilon(),0.0001); // Default eps

		// AMS Grad
		updater = DL4JUtils.toUpdater("amsGrad;;;;.5");
		Assert.assertTrue(updater instanceof AMSGrad);
		Assert.assertEquals(AMSGrad.DEFAULT_AMSGRAD_LEARNING_RATE, ((AMSGrad)updater).getLearningRate(),0.0001);
		Assert.assertEquals(.5, ((AMSGrad)updater).getEpsilon(),0.0001);

		// N adam
		updater = DL4JUtils.toUpdater("nadam;.5");
		Assert.assertTrue(updater instanceof Nadam);
		Assert.assertEquals(.5, ((Nadam)updater).getLearningRate(),0.0001);

		// Nesterovs
		updater = DL4JUtils.toUpdater("nesterovs;.005;0.5");
		Assert.assertTrue(updater instanceof Nesterovs);
		Assert.assertEquals(.005, ((Nesterovs)updater).getLearningRate(),0.0001);
		Assert.assertEquals(.5, ((Nesterovs)updater).getMomentum(),0.0001);

		// No Op
		updater = DL4JUtils.toUpdater("NoOP;.5");
		Assert.assertTrue(updater instanceof NoOp);

		// RMS Prop
		updater = DL4JUtils.toUpdater("RMSProp;.5;.25;");
		Assert.assertTrue(updater instanceof RmsProp);
		Assert.assertEquals(.5, ((RmsProp)updater).getLearningRate(),0.0001);
		Assert.assertEquals(.25, ((RmsProp)updater).getRmsDecay(),0.0001);
	}
	
	@Test
	public void testReverseWeightInit() {
		Assert.assertTrue(DL4JUtils.reverseLookup(WeightInit.XAVIER.getWeightInitFunction()).equals(WeightInit.XAVIER));
	}

}

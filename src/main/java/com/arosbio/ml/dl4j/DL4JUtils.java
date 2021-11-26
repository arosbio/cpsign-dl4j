package com.arosbio.ml.dl4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
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

import com.arosbio.commons.TypeUtils;

public class DL4JUtils {

	private static final String LINE_SEP = System.lineSeparator();
	public static final String UPDATER_SUB_PARAM_SPLITTER = ";";

	private static final String SGD_NAME ="SGD";
	private static final String ADA_BELEIF_NAME = "AdaBelief";
	private static final String ADA_DELTA_NAME = "AdaDelta";
	private static final String ADA_GRAD_NAME = "AdaGrad";
	private static final String ADAM_NAME = "Adam";
	private static final String ADA_MAX_NAME = "AdaMax";
	private static final String AMS_GRAD_NAME = "AMSGrad";
	private static final String N_ADAM_NAME = "Nadam";
	private static final String NESTEROVS_NAME = "Nesterovs";
	private static final String NO_OP_NAME = "NoOp";
	private static final String RMS_PROP_NAME = "RmsProp";

	private static class InvalidSyntax extends RuntimeException {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		final String algName;
		final String expectedSyntax;
		public InvalidSyntax(String name,String expectedSyntax) {
			this.algName = name;
			this.expectedSyntax = expectedSyntax;
		}
	}

	public static IUpdater toUpdater(String input) {
		if (input == null || input.trim().isEmpty())
			throw new IllegalArgumentException("Invalid updater <null>");
		try {
			String lc = input.toLowerCase(Locale.ENGLISH).trim();
			if (lc.startsWith("\"") && lc.endsWith("\"")) {
				lc = lc.substring(1, lc.length()-1);
			}
			lc += ' '; // Add a space in the end, so the splitting of UPDATER_SUB_PARAM_SPLITTER will not remove the last index
			String name = lc.split(UPDATER_SUB_PARAM_SPLITTER)[0].trim();
			List<Double> args = getArgs(lc);
			// Cases for each implementation
			if (SGD_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new Sgd();
				else if (args.size()==1)
					return new Sgd(getOrDefault(args, 0, Sgd.DEFAULT_SGD_LR));
				else {
					throw new InvalidSyntax(SGD_NAME,sgdSyntax());
				}
			} else if (ADA_BELEIF_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new AdaBelief();
				else if (args.size()<=4)
					return new AdaBelief(getOrDefault(args, 0, AdaBelief.DEFAULT_LEARNING_RATE), // LR
							getOrDefault(args, 1, AdaBelief.DEFAULT_BETA1_MEAN_DECAY), // beta1
							getOrDefault(args, 2, AdaBelief.DEFAULT_BETA2_VAR_DECAY), // beta2
							getOrDefault(args, 3, AdaBelief.DEFAULT_EPSILON) // epsilon
							);
				else 
					throw new InvalidSyntax(ADA_BELEIF_NAME,adaBeleifSyntax());
			} else if (ADA_DELTA_NAME.equalsIgnoreCase(name)){
				if (args.isEmpty())
					return new AdaDelta();
				else if (args.size() <=2)
					return new AdaDelta(getOrDefault(args, 0, AdaDelta.DEFAULT_ADADELTA_RHO), // Rho
							getOrDefault(args, 1, AdaDelta.DEFAULT_ADADELTA_EPSILON)); // Epsilon
				else 
					throw new InvalidSyntax(ADA_DELTA_NAME, adaDeltaSyntax());
			} else if (ADA_GRAD_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new AdaGrad();
				else if (args.size()<=2)
					return new AdaGrad(getOrDefault(args, 0, AdaGrad.DEFAULT_ADAGRAD_LEARNING_RATE), // LR 
							getOrDefault(args, 1, AdaGrad.DEFAULT_ADAGRAD_EPSILON)); // Epsilon
				else
					throw new InvalidSyntax(ADA_GRAD_NAME, adaGradSyntax());
			} else if (ADAM_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new Adam();
				else if (args.size() <= 4)
					return new Adam(getOrDefault(args, 0, Adam.DEFAULT_ADAM_LEARNING_RATE), // LR
							getOrDefault(args, 1, Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY), // Beta1
							getOrDefault(args, 2, Adam.DEFAULT_ADAM_BETA2_VAR_DECAY), // Beta2
							getOrDefault(args, 3, Adam.DEFAULT_ADAM_EPSILON)); // Epsilon
				else 
					throw new InvalidSyntax(ADAM_NAME, adamSyntax());
			} else if (ADA_MAX_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new AdaMax();
				else if (args.size() <= 4)
					return new AdaMax(getOrDefault(args, 0, AdaMax.DEFAULT_ADAMAX_LEARNING_RATE), // LR
							getOrDefault(args, 1, AdaMax.DEFAULT_ADAMAX_BETA1_MEAN_DECAY), // Beta1
							getOrDefault(args, 2, AdaMax.DEFAULT_ADAMAX_BETA2_VAR_DECAY), // Beta2
							getOrDefault(args, 3, AdaMax.DEFAULT_ADAMAX_EPSILON)); // Epsilon
				else 
					throw new InvalidSyntax(ADA_MAX_NAME, adaMaxSyntax());
			} else if (AMS_GRAD_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new AMSGrad();
				else if (args.size() <= 4)
					return new AMSGrad(getOrDefault(args, 0, AMSGrad.DEFAULT_AMSGRAD_LEARNING_RATE), // LR
							getOrDefault(args, 1, AMSGrad.DEFAULT_AMSGRAD_BETA1_MEAN_DECAY), // Beta1
							getOrDefault(args, 2, AMSGrad.DEFAULT_AMSGRAD_BETA2_VAR_DECAY), // Beta2
							getOrDefault(args, 3, AMSGrad.DEFAULT_AMSGRAD_EPSILON)); // Epsilon
				else 
					throw new InvalidSyntax(AMS_GRAD_NAME, amsGradSyntax());
			} else if (N_ADAM_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new Nadam();
				else if (args.size() <= 4)
					return new Nadam(getOrDefault(args, 0, Nadam.DEFAULT_NADAM_LEARNING_RATE), // LR
							getOrDefault(args, 1, Nadam.DEFAULT_NADAM_BETA1_MEAN_DECAY), // Beta1
							getOrDefault(args, 2, Nadam.DEFAULT_NADAM_BETA2_VAR_DECAY), // Beta2
							getOrDefault(args, 3, Nadam.DEFAULT_NADAM_EPSILON)); // Epsilon
				else 
					throw new InvalidSyntax(N_ADAM_NAME, nAdamSyntax());
			} else if (NESTEROVS_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty()) {
					return new Nesterovs();
				}else if (args.size() <= 2)
					return new Nesterovs(getOrDefault(args, 0, Nesterovs.DEFAULT_NESTEROV_LEARNING_RATE), // LR
							getOrDefault(args, 1, Nesterovs.DEFAULT_NESTEROV_MOMENTUM)); // Momentum
				else 
					throw new InvalidSyntax(NESTEROVS_NAME, nesterovsSyntax());
			} else if (NO_OP_NAME.equalsIgnoreCase(name)) {
				if (! args.isEmpty())
					throw new IllegalArgumentException(String.format("Invalid syntax for updater %s, only accepted syntax is '%s', got: %s", NO_OP_NAME, NO_OP_NAME,input));
				return new NoOp();
			} else if (RMS_PROP_NAME.equalsIgnoreCase(name)) {
				if (args.isEmpty())
					return new RmsProp();
				else if (args.size()<=3) {
					return new RmsProp(getOrDefault(args, 0, RmsProp.DEFAULT_RMSPROP_LEARNING_RATE), // LR
							getOrDefault(args, 1, RmsProp.DEFAULT_RMSPROP_RMSDECAY), // Rms Decay
							getOrDefault(args, 2, RmsProp.DEFAULT_RMSPROP_EPSILON)  // Epsilon
							);
				} else {
					throw new InvalidSyntax(RMS_PROP_NAME,rmsPropSyntax());
				}
			} else {
				throw new IllegalArgumentException("Invalid Updater: " + input);
			}
		} catch (InvalidSyntax syntErr) {
			throw new IllegalArgumentException(String.format("Invalid syntax for updater %s, expected syntax: %s",syntErr.algName,syntErr.expectedSyntax));
		} catch(IllegalArgumentException e) {
			throw e;
		} catch (Exception e) {
			throw new IllegalArgumentException("Invalid Updater: " + input);
		}
	}

	private static double getOrDefault(List<Double> lst, int index, double fallback) {
		try {
			Double d = lst.get(index);
			if (d != null) return d;
		} catch (Exception e) {}
		return fallback;
	}

	private static List<Double> getArgs(String input){
		List<Double> args = new ArrayList<>();
		String[] splits = input.split(UPDATER_SUB_PARAM_SPLITTER);
		if (splits.length>1) {
			for (int i=1;i<splits.length; i++) {
				try {
					args.add(TypeUtils.asDouble(splits[i]));
				} catch (Exception e) {
					args.add(null);
				}
			}
		}
		return args;
	}

	private static String sgdSyntax() {
		return String.format("%s or %s;<learning rate>",
				SGD_NAME,SGD_NAME);
	}
	private static String adaBeleifSyntax() {
		return String.format("%s or %s;<learning rate>",
				ADA_BELEIF_NAME,ADA_BELEIF_NAME);
	}
	private static String adaDeltaSyntax() {
		return String.format("%s or %s;<rho>;<epsilon>",
				ADA_DELTA_NAME,ADA_DELTA_NAME);
	}
	private static String adaGradSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<epsilon>",
				ADA_GRAD_NAME,ADA_GRAD_NAME,ADA_GRAD_NAME);
	}
	private static String adamSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<beta1>;<beta2>;<epsilon>",
				ADAM_NAME,ADAM_NAME,ADAM_NAME);
	}
	private static String adaMaxSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<beta1>;<beta2>;<epsilon>",
				ADA_MAX_NAME,ADA_MAX_NAME,ADA_MAX_NAME);
	}
	private static String amsGradSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<beta1>;<beta2>;<epsilon>",
				AMS_GRAD_NAME,AMS_GRAD_NAME,AMS_GRAD_NAME);
	}
	private static String nAdamSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<beta1>;<beta2>;<epsilon>",
				N_ADAM_NAME,N_ADAM_NAME,N_ADAM_NAME);
	}
	private static String nesterovsSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<momentum>",
				NESTEROVS_NAME,NESTEROVS_NAME,NESTEROVS_NAME);
	}
	private static String rmsPropSyntax() {
		return String.format("%s or %s;<learning rate> or %s;<learning rate>;<rms decay>;<epsilon>",
				RMS_PROP_NAME,RMS_PROP_NAME,RMS_PROP_NAME);
	}

	public static String toString(IUpdater updater) {
		if (updater instanceof Sgd) {
			Sgd u = (Sgd) updater;
			return SGD_NAME+UPDATER_SUB_PARAM_SPLITTER+u.getLearningRate();
		} else if (updater instanceof AdaBelief) {
			AdaBelief u = (AdaBelief) updater;
			return toNiceString(ADA_BELEIF_NAME,u.getLearningRate(), u.getBeta1(), u.getBeta2(), u.getEpsilon());
		} else if (updater instanceof AdaDelta) {
			AdaDelta u = (AdaDelta) updater;
			return toNiceString(ADA_DELTA_NAME,u.getRho(),u.getEpsilon());
		} else if (updater instanceof AdaGrad) {
			AdaGrad u = (AdaGrad) updater;
			return toNiceString(ADA_GRAD_NAME,u.getLearningRate(),u.getEpsilon());
		} else if (updater instanceof Adam) {
			Adam u = (Adam)updater;
			return toNiceString(ADAM_NAME,u.getLearningRate(), u.getBeta1(), u.getBeta2(), u.getEpsilon());
		} else if (updater instanceof AdaMax) {
			AdaMax u = (AdaMax)updater;
			return toNiceString(ADA_MAX_NAME, u.getLearningRate(), u.getBeta1(), u.getBeta2(), u.getEpsilon());
		} else if (updater instanceof AMSGrad) {
			AMSGrad u = (AMSGrad) updater;
			return toNiceString(AMS_GRAD_NAME, u.getLearningRate(), u.getBeta1(), u.getBeta2(), u.getEpsilon());
		} else if (updater instanceof Nadam) {
			Nadam u = (Nadam) updater;
			return toNiceString(N_ADAM_NAME, u.getLearningRate(), u.getBeta1(), u.getBeta2(), u.getEpsilon());
		} else if (updater instanceof Nesterovs) {
			Nesterovs u = (Nesterovs) updater;
			return toNiceString(NESTEROVS_NAME, u.getLearningRate(), u.getMomentum());
		} else if (updater instanceof NoOp) {
			return NO_OP_NAME;
		} else if (updater instanceof RmsProp) {
			RmsProp u = (RmsProp) updater;
			return toNiceString(RMS_PROP_NAME, u.getLearningRate(),u.getRmsDecay(),u.getEpsilon());
		} 
		throw new IllegalArgumentException("Unsupported updater: "+updater.toString());
	}

	private static String toNiceString(Object... nameAndParams) {
		StringBuilder sb=new StringBuilder();
		for (int i=0;i<nameAndParams.length-1;i++)
			sb.append(nameAndParams[i]).append(UPDATER_SUB_PARAM_SPLITTER);
		sb.append(nameAndParams[nameAndParams.length-1]);
		return sb.toString();
	}

	public static String supportedUpdaterInput() {
		StringBuilder sb = new StringBuilder();
		sb.append(sgdSyntax()).append(LINE_SEP);
		sb.append(adaBeleifSyntax()).append(LINE_SEP);
		sb.append(adaDeltaSyntax()).append(LINE_SEP);
		sb.append(adaGradSyntax()).append(LINE_SEP);
		sb.append(adamSyntax()).append(LINE_SEP);
		sb.append(adaMaxSyntax()).append(LINE_SEP);
		sb.append(amsGradSyntax()).append(LINE_SEP);
		sb.append(nAdamSyntax()).append(LINE_SEP);
		sb.append(nesterovsSyntax()).append(LINE_SEP);
		sb.append(NO_OP_NAME).append(LINE_SEP);
		sb.append(rmsPropSyntax());
		return sb.toString();
	}

	public static WeightInit reverseLookup(IWeightInit fun) {
		for (WeightInit v : WeightInit.values()) {
			try {
				if (v.getWeightInitFunction().getClass().equals(fun.getClass())) {
					return v;
				}
			} catch (Exception e) {
				// In case the weight init function requires some additional parameters
			}
		}
		throw new IllegalArgumentException("Could not find the correct weightInit function for class: " + fun.getClass().getName());
	}

	public static Activation reverseLookup(IActivation fun) {
		for (Activation v : Activation.values()) {
			try {
				if (v.getActivationFunction().getClass().equals(fun.getClass())) {
					return v;
				}
			} catch (Exception e) {
				// In case the activation function requires some additional parameters
			}
		}
		throw new IllegalArgumentException("Could not find the correct activation function for class: " + fun.getClass().getName());
	}
}

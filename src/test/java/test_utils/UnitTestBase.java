package test_utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import com.arosbio.data.DataRecord;
import com.arosbio.data.Dataset;
import com.arosbio.data.Dataset.SubSet;
import com.arosbio.data.FeatureVector.Feature;
import com.arosbio.data.NamedLabels;

public class UnitTestBase {
	
	// Classification data sets
	
	/** The well-known Iris data set with 3 classes, 150 examples and 4 features */
	public static final String IRIS_REL_PATH = "/data/class/iris.svmlight.gz";
	/** 
	 * The UCI dataset over Forest Cover type - scaled features to [0..1] range. 
	 * 581012 examples, 54 features, 7 classes
	 */
	public static final String COV_TYPE_REL_PATH = "/data/class/covtype.scale01.gz";
	/** 100 examples and 57 features. Features 0-53 all fairly small, 54-56 much larger range */
	public static final String BINARY_REL_PATH = "/data/class/train.svmlight.gz";
	
	/** UCI data set Mandelon, 2000 examples, 500 features. 1000 of two classes */
	public static final String MADELON_REL_PATH = "/data/class/madelon.train.gz";
	
	/** Ames data set with chemical structures */
	public static class AmesBinaryClf {
		public static final String AMES_REL_PATH = "/data/class/ames_small.sdf.gz";
		public static final String PROPERTY = "Ames test categorisation";
		public static final NamedLabels LABELS = new NamedLabels(Arrays.asList("mutagen", "nonmutagen"));
	}
	
	// Regression data
	
	/** Androgen receptor data, 688 examples and 97 features. Need scaling and some features are all 0 */
	public static final String ANDROGEN_REL_PATH = "/data/reg/androgen.svmlight.gz";
	
	public static final String ENRICH_REL_PATH = "/data/reg/enrich.svmlight.gz";
	
	public static Dataset.SubSet getIrisClassificationData() throws IOException {
		try (InputStream is = UnitTestBase.class.getResourceAsStream(IRIS_REL_PATH)){
			return Dataset.fromLIBSVMFormat(is).getDataset();
		}
	}
	
	public static Dataset.SubSet getCoverTypeClassificationData() throws IOException{
		try (InputStream is = UnitTestBase.class.getResourceAsStream(COV_TYPE_REL_PATH)){
			return Dataset.fromLIBSVMFormat(is).getDataset();
		}
	}
	
	public static Dataset.SubSet getBinaryClassificationData() throws IOException{
		try (InputStream is = UnitTestBase.class.getResourceAsStream(BINARY_REL_PATH)){
			return Dataset.fromLIBSVMFormat(is).getDataset();
		}
	}
	
	public static Dataset.SubSet getMadelonRegressionData() throws IOException{
		try (InputStream is = UnitTestBase.class.getResourceAsStream(MADELON_REL_PATH)){
			return Dataset.fromLIBSVMFormat(is).getDataset();
		}
	}
	
	
	public static Dataset.SubSet getAndrogenReceptorRegressionData() throws IOException{
		try (InputStream is = UnitTestBase.class.getResourceAsStream(ANDROGEN_REL_PATH)){
			return Dataset.fromLIBSVMFormat(is).getDataset();
		}
	}
	
	public static Dataset.SubSet getEnrichmentRegressionData() throws IOException{
		try (InputStream is = UnitTestBase.class.getResourceAsStream(ENRICH_REL_PATH)){
			return Dataset.fromLIBSVMFormat(is).getDataset();
		}
	}
	
	
	
//	@Test
	public void testLoad() throws IOException {
		SubSet d = getEnrichmentRegressionData();
		System.err.println(d.toString());
//		System.err.println(d.getLabelFrequencies());
		
		SummaryStatistics labels = new SummaryStatistics();
		List<SummaryStatistics> ssList = new ArrayList<>();
		
		for (DataRecord r : d) {
			for (Feature f : r.getFeatures()) {
				while (f.getIndex() + 1 > ssList.size()) {
					ssList.add(new SummaryStatistics());
				}
				ssList.get(f.getIndex()).addValue(f.getValue());
			}
			labels.addValue(r.getLabel());
		}
		System.err.println("Label: " + labels.toString());
		
		for (int i=0; i<ssList.size(); i++) {
			SummaryStatistics ss = ssList.get(i);
			System.out.println("" + i + " ["+ss.getMin() + " .. "+ss.getMax()+"] " + ss.getN());
		}
	}

	
	
	
}

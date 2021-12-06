package com.arosbio.ml.dl4j;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Enumeration;
import java.util.Properties;

public class MainClass {

	private final static String PROPERTIES_FILE = "cpsign-dl4j.properties";
	private final static String BUILD_VERSION_KEY = "build.version";
	private final static String TIME_STAMP_KEY = "build.ts";

	public static void main(String[] args) {

		// Get the build version and timestamp 
		String v = null, ts = null;
		Properties p = new Properties();
		try {
			Enumeration<URL> resources = MainClass.class.getClassLoader()
					.getResources(PROPERTIES_FILE);
			if (resources.hasMoreElements()) {
				try (InputStream is = resources.nextElement().openStream();){
					p.load(is);
					v = p.getProperty(BUILD_VERSION_KEY, "Undefined");
					ts = p.getProperty(TIME_STAMP_KEY, "Undefined");
				} catch(IOException ioe) {} 
			}
		} catch (Exception e){}

		StringBuilder sb = new StringBuilder();
		sb.append("%nThis JAR file is an extension to the CPSign program - providing support for Deeplearning based ML models.%n");
		if (v != null)
			sb.append("Version: ").append(v).append("%n");
		if (ts != null)
			sb.append("Build time: ").append(ts).append("%n");
		sb.append("Please visit https://github.com/arosbio/cpsign-dl4j for more information%n%n");
		
		
		System.out.printf(sb.toString());
	}

}

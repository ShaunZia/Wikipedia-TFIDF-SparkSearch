# Import SparkConf & SparkContext libraries to run Python script
# Import MLLib's HashingTF & IDF libraries to compute TF & IDF
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Create a local Spark configuration and Spark context
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line) by creating a Spark context (sc)
# Spark context references a sub-sample of Wiki articles from subset-small.tsv
# tsv stands for tab-separated values
# Split tsv document into a Python list of tab-delimited fields
# Extract field 3, containing the actual text, and split it into a list of words  
rawData = sc.textFile("C:\MachineLearningFrankKane\DataScience\subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names in field 1
documentNames = fields.map(lambda x: x[1])

# Compute term frequencies by hashing every word in each document into 1 of
# 100,000 numerical values, in order to save memory
# This technique strips out missing data, thus creating an RDD of sparse vectors
hashingTF = HashingTF(100000)  
tf = hashingTF.transform(documents)

# Cache the TF RDD for reuse
# Ignore any word that doesn't appear at least twice in the TF and IDF
# Create an RDD of sparse vectors, comprised of unique TF-IDF hash values
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Search for "Gettysburg", expecting it to appear in "Abraham Lincoln"
# Transform "Gettysburg" into its hash value and find its sparse vector index
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Extract the TF-IDF hash value for "Gettysburg" into a new relevance RDD 
# for each document
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# Combine relevance with document names using the zip function
zippedResults = gettysburgRelevance.zip(documentNames)

# Print the document with the maximum TF*IDF value
print "Best document for Gettysburg is:"
print zippedResults.max()

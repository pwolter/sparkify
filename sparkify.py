#Install pandas and matplotlib
sc.install_pypi_package("pandas")
sc.install_pypi_package("matplotlib")

# All libraries required fo rthis analysis
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, NaiveBayes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setting to display max column width
pd.set_option('display.max_colwidth', None)

# conf = spark.sparkContext._conf.setAll([
#     ('spark.executor.memory', '8g'), 
#     ('spark.executor.cores', '4'), 
#     ('spark.cores.max', '4'), 
#     ('spark.driver.memory','6g'),
#     ('spark.default.parallelism', '16')
# ])

# Create a Spark session
spark = SparkSession.builder \
        .appName('Sparkify') \
        .getOrCreate()

# Cerate a Dataframe reading the JSON file
# data_file = "s3n://pwolter-notebook-work/medium-sparkify-event-data.json"
data_file = "s3n://udacity-dsnd/sparkify/medium-sparkify-event-data.json"
df = spark.read.json(data_file)

# I check the head of the dataframe
df.head()

# print the dataset schema to check columns types
df.printSchema()

# df = all_df.sample(withReplacement=False, fraction=0.2, seed=1234)
# df = all_df.select("*")

df = df.withColumn("userId", df["userId"].cast(IntegerType()))

# Let's print total number of rows and columns
print("Total number of rows: {}\nNumber of columns: {}".format(df.count(), len(df.columns)))

# I first create a second dataframe with only the 
# columns required for feature building
columns_to_include = ['auth', 'gender', 'length', 'level', 'location', 'page', 'sessionId', 'song', 'status', 'ts', 'userAgent', 'userId']

# We filter and persist the dataframe
df_new = df.select(columns_to_include).persist()

# I create the 'churn' column defining churn as all events of 'Cancellation Confirmation' and 'Submit Downgrade'
# from teh 'page' column as these indicate a clear intention of the user to either cancel teh subscription
# or to downgrade it. So 'churn' equals '1' and 'no-churn' equals '0'
df_new = df_new.withColumn('churn', when(df_new['page'] == 'Cancellation Confirmation', 1).otherwise(0))
df_new = df_new.withColumn('churn', when(df_new['page'] == 'Submit Downgrade', 1).otherwise(df_new['churn']))
df_new = df_new.withColumn("churn", df_new["churn"].cast(IntegerType()))

# We can see that the 'churn' column is very unbalanced
df_new.groupby('churn').count().sort('count', ascending=[False]).show()

# Based on the previous information I am exploring some columns I think
# are interesting or may add value to the prediction of churn
df_new.groupby('auth').count().sort('count', ascending=False).show()

# I compute the distribution of gender
df_new.groupby('gender').count().sort('count', ascending=False).show()

# We have some users that ddi not provide a gender so I label it 'U'
df_new = df_new.withColumn('gender', when(isnull('gender'), 'U').otherwise(df_new['gender']))

# I graph it
plt.clf()
df_new.groupby('gender').count().sort('count', ascending=False) \
.toPandas().plot.bar(x='gender', y='count', rot=0, legend=None, cmap = 'RdYlBu')
plt.title("Gender distribution of users")
plt.xlabel("Gender")
plt.ylabel("Number of users")
plt.tight_layout()
%matplot plt

# Let's see the 'level' distribution of subscribers
df_new.groupby('level').count().sort('level', ascending=[False]).show()

# A pie chart makes sense here, most of our users are paid users
labels = ['paid', 'free']
plt.clf()
df_new.groupby('level').count().sort('level', ascending=[False]) \
.toPandas().plot.pie(y='count', labels=labels, cmap = 'RdYlBu');
plt.ylabel("")
plt.title("Distribution of paid and free users")
plt.tight_layout()
%matplot plt

# Let's check the 'page' column
df_new.groupby('page').count().sort('count', ascending=False).show()

# The same in a graphical representation
plt.clf()
df_new.groupby('page').count().sort('count', ascending=True) \
.toPandas().plot.barh(x='page', y='count', rot=0, legend=None, cmap = 'RdYlBu')
plt.title("Page distribution")
plt.xlabel("Page hits")
plt.ylabel("")
plt.tight_layout()
%matplot plt

# We have 3 status: 200, 307 and 404
# the last two are failures 
df_new.groupby('status').count().sort('count', ascending=[False]).show()

# Source idea from here: https://medium.com/@junwan01/oversampling-and-undersampling-with-pyspark-5dbc25cdf253
major_df = df_new.where(df_new["churn"] == 0)
minor_df = df_new.where(df_new["churn"] == 1)

major_df_count = major_df.count()
minor_df_count = minor_df.count()

ratio = int(major_df_count/minor_df_count)
print("ratio: {}".format(ratio))

a = range(ratio)

# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')

# combine both oversampled minority rows and previous majority rows 
df_new = major_df.unionAll(oversampled_df)

print("df count before resample: {}".format(df.count()))

print("df count after resample: {}".format(df_new.count()))

# We can see that the 'churn' column is very less unbalanced now
df_new.groupby('churn').count().sort('count', ascending=[False]).show()

# I use a StringIndexer in the 'churn' column and named it 'label' as
# required by the Spark ML library in the respective section
churnIndexer = StringIndexer(inputCol="churn", outputCol="label")

# Transform teh column
df_new = churnIndexer.fit(df_new).transform(df_new)

# And delete the 'churn' column as I don't need it anymore
df_new = df_new.drop('churn')

# I apply a StringIndexer in this column as well
genderIndexer = StringIndexer(inputCol="gender", outputCol="gender_int")

# Transform it
df_new = genderIndexer.fit(df_new).transform(df_new)

# and delete the categorical
df_new = df_new.drop('gender')

# StringIndexer applied
levelIndexer = StringIndexer(inputCol="level", outputCol="level_int")

# Transform
df_new = levelIndexer.fit(df_new).transform(df_new)

# Column deleted
df_new = df_new.drop('level')

# let's see what the 'location' column has to offer:
df_new.groupby('location').count().sort('count', ascending=[False]).show(20, False)

# The US, according to 'www2.census.gov', can be divided in 4 mayor regions: Region 1: Northeast, 
# Region 2: Midwest, Region 3: South and Region 4: West.
# Each of these regions can be divided further, for example, Region 1 is compromised of Division 1: 
# New England with Connecticut, Maine, Massachusetts, New Hampshire, RhodeIsland and, Vermont and so for.
# I just divided it into 4 Regions: Northeast, Midwest, South and West.
location_regions = {
    'Northeast': ['NJ', 'NY', 'PA', 'CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
    'Midwest': ['IN', 'IL', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'South': ['DE', 'DC', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
    'West': ['AZ', 'CO', 'ID', 'NM', 'MT', 'UT', 'NV', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
}

# Extract the State information from the 'location' column
df_new = df_new.withColumn('state', split(df_new['location'], ',').getItem(1))

# Drop 'location' column as I don't need it anymore
df_new = df_new.drop('location')

# A quick check on our newly created column 
df_new.select('state').distinct().sort('state').show(10, False)

# let's check the 'userAgent' column
df_new.select('userAgent').show(5, False)

# I extrtact the 'os' (Operating System) information as it may play a role in churn
df_new = df_new.withColumn('os', regexp_extract(col('userAgent'), r'\((\w+);?\s+', 1))

# I also, extract the browser from the 'userAgent' column 
df_new = df_new.withColumn('browser', regexp_extract(col('userAgent'), r'\s(\w+)/\d+.\d+"?$', 1))

# A summary that revealed 'empty' fields
df_new.groupby('browser').count().sort('count', ascending=[False]).show()

# I first take care of nulls/nan filling them with 'Undefined'
df_new = df_new.fillna('Undefined', subset='browser')

# I do the same with the 'empty' values as I am not sure what this really mean in this context
# maybe here the opinion of a subject matter expert will be beneficial or helpfull
df_new = df_new.withColumn('browser', when(df_new['browser'] == '', 'Undefined').otherwise(df_new['browser']))

# The data looks fixed now
df_new.groupby('browser').count().sort('count', ascending=[False]).show()

# I do take care of them for the os as well filling them with 'Undefined'
df_new = df_new.fillna('Undefined', subset='os')

# And all nulls corrected
df_new.groupby('os').count().sort('count', ascending=[False]).show()

# StringIndexer applied for both 'os' and 'browser'
osIndexer = StringIndexer(inputCol="os", outputCol="os_int")

df_new = osIndexer.fit(df_new).transform(df_new)

browserIndexer = StringIndexer(inputCol="browser", outputCol="browser_int")

df_new = browserIndexer.fit(df_new).transform(df_new)

# Columns deleted as I do not need them anymore
df_new = df_new.drop(*['os', 'browser'])

# I chech the 'ts' (timestamp) column
df_new.select(df_new.ts, length(df_new.ts)).show(5, False)

# We do have the 'ts' in miliseconds, as it is 13 digits long, so we first need to 
# convert it to seconds dividing it by 1000. Then convert it to a date
df_new = df_new.withColumn('ts_date', from_unixtime(col('ts') / 1000))

# From the new 'ts_date' column I extract the day of the week and the hour of the day
# 0 - Sunday, 1 - Monday, etc.
df_new = df_new.withColumn('day_of_week', dayofweek(col('ts_date')))
# Integer value for the hour
df_new = df_new.withColumn('hour', hour(col('ts_date')))

# Then I calculate the songs per hour
songs_per_hour = df_new.filter(df_new.page == "NextSong")\
                .groupby(df_new.hour).count()\
                .orderBy(df_new.hour.cast("float")).toPandas()

# And how many songs are played by day of the week
songs_per_day_of_week = df_new.filter(df_new.page == "NextSong")\
                .groupby(df_new.day_of_week).count()\
                .orderBy(df_new.day_of_week.cast("float")).toPandas()

songs_per_hour.hour = pd.to_numeric(songs_per_hour.hour)

plt.clf()
plt.scatter(songs_per_hour["hour"], songs_per_hour["count"], color='darkred')
plt.xticks(np.arange(0, 26, 2))
plt.title("Number of songs played by hour of day [0-24]")
plt.xlabel("Hour of day")
plt.ylabel("Songs played")
plt.tight_layout()
%matplot plt

songs_per_day_of_week.day_of_week = pd.to_numeric(songs_per_day_of_week.day_of_week)
labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

plt.clf()
plt.bar(songs_per_day_of_week["day_of_week"], 
        songs_per_day_of_week["count"], 
        tick_label=labels, color='darkred')
plt.xlim(0, 8)
plt.title("Number of songs played by day of the week")
plt.xlabel("Day of week")
plt.ylabel("Songs played")
plt.tight_layout()
%matplot plt

# I define some features that define users interaction with the service.
# Here I capture the user saving settings as a measure of interaction
df_1 = df_new.select('userId', 'page')\
        .where('page = "Save Settings"')\
        .groupby('userId').count()\
       .withColumnRenamed('count', 'saved_settings')

# then I merge the column into my dataframe
df_new = df_new.join(df_1, 'userId', 'inner')

# calculate songs played by user
df_2 = df_new.select('userId', 'song')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'num_songs')

df_new = df_new.join(df_2, 'userId', 'inner')

# Thums up songs means a positive interaction with the syetm
# so I capture it
df_3 = df_new.select('userId', 'page')\
        .where('page = "Thumbs Up"')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'thumbs_up')

df_new = df_new.join(df_3, 'userId', 'inner')

# For free users too many advertising could result in
# a poor perceived service, so I capture that as a feature
df_4 = df_new.select('userId', 'page')\
        .where('page = "Roll Advert"')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'num_advertisement')

df_new = df_new.join(df_4, 'userId', 'inner')

# I thouhg conbining this with 'thumbs_up' as an indication of
# interaction of users with the service. I kept it separate as
# this particular feature may indicate that the recommended songs
# for the user are not good and hence he thumbs them down and maybe
# factors in him/her churning
df_5 = df_new.select('userId', 'page')\
        .where('page = "Thumbs Down"')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'thumbs_down')

df_new = df_new.join(df_5, 'userId', 'inner')

# If users are adding lots of songs to their playlist
# that may indicate a user that wants to stay or convert 
# from free to paid, consequently not churning
df_6 = df_new.select('userId', 'page')\
        .where('page = "Add to Playlist"')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'playlist_added')

df_new = df_new.join(df_6, 'userId', 'inner')

# Adding friends may mean something positive
df_7 = df_new.select('userId', 'page')\
        .where('page = "Add Friend"')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'friend_added')

df_new = df_new.join(df_7, 'userId', 'inner')

# Errors are bad experience with the service so we need to
# capture that
df_8 = df_new.select('userId', 'page')\
        .where('page = "Error"')\
        .groupby('userId').count()\
        .withColumnRenamed('count', 'errors_pages')

df_new = df_new.join(df_8, 'userId', 'inner')

# A user that listens lots of songs per session indicate
# users that are happy with the service
df_9 = df_new.where('page == "NextSong"') \
        .groupby(['userId', 'sessionId']) \
        .count() \
        .groupby(['userId']) \
        .agg({'count':'avg'}) \
        .withColumnRenamed('avg(count)', 'songs_persession')

df_new = df_new.join(df_9, 'userId', 'inner')

# I drop all the columns I do not need any longer to train my models
df_new = df_new.drop(*['ts_date', 'state', 'userAgent', 'ts', 'status', 'song', 'sessionId', 'page', 'length', 'auth', 'userId'])

# I define the 'features' column to use in the model
feature_columns = ['gender_int', 'level_int', 'os_int', 'browser_int', 'day_of_week', 
                   'hour', 'saved_settings', 'num_songs', 'thumbs_up', 'num_advertisement', 
                   'thumbs_down', 'playlist_added', 'friend_added', 'errors_pages', 'songs_persession']

# I use VectorAssembler to vectorize the features
assembler = VectorAssembler(inputCols=feature_columns, outputCol='vector_features')

# I transform the column
df_new = assembler.transform(df_new)

# And normalize the data to avoid bias with higuer values in the feature vector
normalizer = Normalizer(inputCol='vector_features', outputCol='features')

# Transform the vector
df_new = normalizer.transform(df_new)

df_new.coalesce(1)
    .write
    .format('json')
    .mode('overwrite')
    .save("s3n://pwolter-notebook-work/sparkify_cleaned_dataset.json")

# Define the seed to use
seed = 1234

# The evaluator which is common to all models
evaluator = MulticlassClassificationEvaluator(labelCol='label')

def confussion_matrix(predictor):
    """Receives a model's predictions dataframe and calculates the 
    precission and recall. Also prints out the true and false positive/negative
    values.
    Args:
    predictor:
        The predictions dataframe used to calculate precission and recall.
    Returns:
        Nothing
    """
    true_negative = predictor.select("*").where("prediction = 0 AND label = 0").count()
    true_positive = predictor.select("*").where("prediction = 1 AND label = 1").count()
    
    false_negative = predictor.select("*").where("prediction = 0 AND label = 1").count()
    false_positive = predictor.select("*").where("prediction = 1 AND label = 0").count()
    
    if (true_positive + false_positive) != 0:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    else:
        precision = 0
        recall = 0
        
    print("True Negative: {}\nTrue Positive: {}".format(true_negative, true_positive))
    print("False Negative: {}\nFalse Positive: {}".format(false_negative, false_positive))
    print("Precission: {}\nRecall: {}".format(precision, recall))

def evaluation_metrics(evaluator, predictions):
    """Receives a model's evaluator and predictions dataframe and calculates 
    f-1, weighted precision, weighted recall and accuracy and prints them
    out.
    Args:
    evaluator:
        The evaluator used to calculate model metrics.
    predictor:
        The predictions dataframe used to calculate precission and recall.
    Returns:
        Nothing
    """
    f1 = evaluator.evaluate(predictions)
    wp = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    wr = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    
    print("F1:\t{}\nWP:\t{}\nWR:\t{}\nAccu:\t{}".format(f1, wp, wr, accuracy))

# Split data set into train and test
train, test = df_new.randomSplit([0.8, 0.2], seed=seed)

# lr = LogisticRegression()

# lr_model = lr.fit(train)

# lr_predictions = lr_model.transform(test)

# lr_predictions.printSchema()

# evaluation_metrics(evaluator, lr_predictions)



# rf = RandomForestClassifier(featureSubsetStrategy='auto', maxDepth=2, seed=seed)

# rf_model = rf.fit(train)

# rf_predictions = rf_model.transform(test)

# evaluation_metrics(evaluator, rf_predictions)



# gbt = GBTClassifier(labelCol='label', maxDepth=2, seed=seed)

# gbt_model = gbt.fit(train)

# gbt_predictions = gbt_model.transform(test)

# evaluation_metrics(evaluator, gbt_predictions)



# nb = NaiveBayes()

# nb_model = nb.fit(train)

# nb_predictions = nb_model.transform(test)

# evaluation_metrics(evaluator, nb_predictions)



gbt = GBTClassifier(labelCol='label', seed=seed)

paramGrid = ParamGridBuilder() \
                    .addGrid(gbt.maxBins, [32]) \
                    .addGrid(gbt.maxDepth, [2, 4]) \
                    .addGrid(gbt.maxIter, [10]) \
                    .build()

pipeline = Pipeline(stages=[gbt])

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, 
                    estimatorParamMaps=paramGrid, numFolds=5)

best_model = cv.fit(train)

predictions = best_model.transform(test)

predictions.printSchema()

evaluation_metrics(evaluator, predictions)

# confussion_matrix(predictions)

predictions.select("RawPrediction","prediction") \
.orderBy("RawPrediction", ascending=True) \
.show(25, False)

best_model.bestModel.stages[0].featureImportances

best_model.bestModel.stages[0].extractParamMap()

# true_negative = predictions.select("*").where("prediction = 0 AND label = 0").count()
# true_positive = predictions.select("*").where("prediction = 1 AND label = 1").count()
# print("True Negative: {}\nTrue Positive: {}".format(true_negative, true_positive))
# Our test predicted 21 customers leaving who actually did leave and also
# predicted 55529 customers not leaving who actually did not leave.

# false_negative = predictions.select("*").where("prediction = 0 AND label = 1").count()
# false_positive = predictions.select("*").where("prediction = 1 AND label = 0").count()
# print("False Negative: {}\nFalse Positive: {}".format(false_negative, false_positive))
# Our test predicted 0 customers leaving who actually did not leave and
# also predicted 0 customers not leaving who actually did leave.


